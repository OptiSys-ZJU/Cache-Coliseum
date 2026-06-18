"""
TrieParrotModel: Tree-state aware cache eviction predictor.

Candidate leaf paths are encoded by Path-LSTM and used as attention queries
over a recent memory of touched trie paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any

from model.trie_model.tree_lstm import PathLSTMCell
from model.trie_model.embed import NodeEmbedder


class TrieParrotModel(nn.Module):
    """
    Tree-state aware predictor for cache eviction.

    Differs from original EvictionPolicyModel in that:
    - Candidate representations come from Tree-LSTM encoded trie paths
    - Supports incremental node state computation
    - History is represented as a short memory bank of recent hidden states
    """

    def __init__(
        self,
        vocab_size: int,
        node_embed_dim: int = 64,
        history_embed_dim: int = 64,
        hidden_size: int = 128,
        max_attention_history: int = 30,
        ranking_loss_weight: float = 1.0,
        reuse_loss_weight: float = 0.1,
        ce_loss_weight: float = 0.0,
        ce_target_policy: str = "argmax",
        candidate_scorer_mode: str = "history_only",
        reuse_distance_log_cap: float = 5.0,
        ndcg_alpha: float = 10.0,
    ):
        """
        Initialize TrieParrotModel.

        Args:
            vocab_size: Size of node ID vocabulary
            node_embed_dim: Dimension of node embeddings (for tree paths)
            history_embed_dim: Dimension of history access embeddings
            hidden_size: Dimension of LSTM hidden states (both history and tree)
            max_attention_history: Max number of past history states for attention
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_attention_history = max_attention_history
        self.ranking_loss_weight = ranking_loss_weight
        self.reuse_loss_weight = reuse_loss_weight
        self.ce_loss_weight = ce_loss_weight
        if ce_target_policy not in {"argmax", "top_set"}:
            raise ValueError(
                "ce_target_policy must be one of {'argmax', 'top_set'}, "
                f"got {ce_target_policy!r}"
            )
        self.ce_target_policy = ce_target_policy
        if candidate_scorer_mode not in {"history_only", "candidate_history_concat"}:
            raise ValueError(
                "candidate_scorer_mode must be one of "
                "{'history_only', 'candidate_history_concat'}, "
                f"got {candidate_scorer_mode!r}"
            )
        self.candidate_scorer_mode = candidate_scorer_mode
        self.reuse_distance_log_cap = reuse_distance_log_cap
        self.ndcg_alpha = ndcg_alpha
        self.last_loss_stats = {
            "full_steps": 0,
            "capped_steps": 0,
            "candidate_count": 0,
        }

        # --- Node embedding (shared between tree path & history) ---
        self.node_embedder = NodeEmbedder(vocab_size, node_embed_dim)

        # Legacy history encoder kept for checkpoint/API compatibility. Trie-PARROT
        # v1 uses path-level history slots encoded by path_lstm instead.
        self.history_lstm = nn.LSTMCell(history_embed_dim, hidden_size)

        # Project node embedding to history input dim (in case they differ)
        if node_embed_dim != history_embed_dim:
            self.history_proj = nn.Linear(node_embed_dim, history_embed_dim)
        else:
            self.history_proj = nn.Identity()

        # --- Path Encoder: Tree-LSTM for encoding root-to-leaf paths ---
        self.path_lstm = PathLSTMCell(node_embed_dim, hidden_size)

        # --- Attention: candidate queries attend over ordered path memory ---
        self.history_pos_embed = nn.Embedding(max_attention_history, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        if candidate_scorer_mode == "candidate_history_concat":
            scorer_input_dim = hidden_size * 2
        else:
            scorer_input_dim = hidden_size
        self.scorer = nn.Linear(scorer_input_dim, 1)

        # Predicts log-capped future request reuse distance.
        self.reuse_distance_estimator = nn.Linear(scorer_input_dim, 1)

    def compute_node_state(
        self,
        node_id: int,
        parent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden state for a single trie node (incremental).

        This is the core method for incremental tree encoding.
        Called when a new node is inserted into the trie.
        The result should be cached in TrieNode.hidden_state.

        Args:
            node_id: Integer ID of the node
            parent_state: Parent's (h, c) state. None for root's children.

        Returns:
            (h, c) tuple, each shape (1, hidden_size)
        """
        device = next(self.parameters()).device
        node_embed = self.node_embedder.embed_single(node_id, device)
        h, c = self.path_lstm(node_embed, parent_state)
        return h, c

    def encode_history_step(
        self,
        node_id: int,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy helper for the pre-v1 token-history LSTM path.

        Trie-PARROT v1 does not call this in runtime or loss replay; history
        slots are path-level vectors produced by _encode_path().

        Args:
            node_id: ID of the currently accessed node
            prev_state: Previous (h, c) from history LSTM. None for first step.

        Returns:
            Updated (h, c) for history LSTM
        """
        device = next(self.parameters()).device
        node_embed = self.node_embedder.embed_single(node_id, device)
        history_input = self.history_proj(node_embed)

        if prev_state is None:
            h = torch.zeros(1, self.hidden_size, device=device)
            c = torch.zeros(1, self.hidden_size, device=device)
            prev_state = (h, c)

        h_new, c_new = self.history_lstm(history_input, prev_state)
        return h_new, c_new

    def _encode_path(
        self,
        path: Optional[Tuple[int, ...]],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a root-to-leaf token path into a single hidden vector."""
        if path is None or len(path) == 0:
            return torch.zeros(1, self.hidden_size, device=device)

        state = None
        for nid in path:
            node_embed = self.node_embedder.embed_single(nid, device)
            state = self.path_lstm(node_embed, state)
        return state[0]

    def _prepare_history_memory(
        self,
        history_memory,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        """Normalize history memory to shape (M, H), preserving whether it is real."""
        if history_memory is None:
            return torch.zeros(1, self.hidden_size, device=device), False
        if isinstance(history_memory, tuple):
            history_memory = history_memory[0]
        if isinstance(history_memory, list):
            if len(history_memory) == 0:
                return torch.zeros(1, self.hidden_size, device=device), False
            history_memory = torch.cat(history_memory, dim=0)
        if history_memory.dim() == 1:
            history_memory = history_memory.unsqueeze(0)
        return history_memory.to(device), history_memory.numel() > 0

    def _add_history_positions(self, history_memory: torch.Tensor) -> torch.Tensor:
        """Add oldest-to-newest positional embeddings to recent history slots."""
        limit = max(1, self.max_attention_history)
        if history_memory.size(0) > limit:
            history_memory = history_memory[-limit:]

        positions = torch.arange(
            history_memory.size(0),
            device=history_memory.device,
            dtype=torch.long,
        )
        return history_memory + self.history_pos_embed(positions)

    def _encode_history_paths(
        self,
        history_paths,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if history_paths is None:
            return None

        paths = [tuple(path) for path in history_paths]
        if not paths:
            return None

        limit = max(1, self.max_attention_history)
        encoded = [self._encode_path(path, device) for path in paths[-limit:]]
        return torch.cat(encoded, dim=0).to(device)

    def forward(
        self,
        history_memory: torch.Tensor,
        candidate_states: List[torch.Tensor] = None,
        candidate_paths: List[Tuple[int, ...]] = None,
        inference: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eviction scores for candidate leaf nodes.

        When inference=True:
            Uses pre-computed candidate_states from trie cache.
        When inference=False:
            Re-encodes candidate_paths from scratch with gradient.

        Args:
            history_memory: Oldest-to-newest path memory, shape (M, hidden_size).
            candidate_states: List of N pre-computed candidate path hidden states.
            candidate_paths: List of N root-to-leaf node ID tuples.
            inference: If True, use cached candidate_states; otherwise re-encode
                candidate_paths with gradient.

        Returns:
            eviction_logits: shape (1, N)
            pred_reuse_distances: shape (1, N)
        """
        device = next(self.parameters()).device
        history_memory, has_history = self._prepare_history_memory(history_memory, device)
        if has_history:
            history_memory = self._add_history_positions(history_memory)

        if inference:
            assert candidate_states is not None, "candidate_states required when inference=True"
            if len(candidate_states) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            candidates = torch.cat(candidate_states, dim=0)
        else:
            assert candidate_paths is not None, "candidate_paths required when inference=False"
            if len(candidate_paths) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            encoded = [self._encode_path(path, device) for path in candidate_paths]
            candidates = torch.cat(encoded, dim=0)

        queries = self.query_proj(candidates)
        memory_keys = self.key_proj(history_memory)

        scale = self.hidden_size ** 0.5
        attn_logits = torch.matmul(queries, memory_keys.T) / scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        retrieved_history = torch.matmul(attn_weights, history_memory)

        if self.candidate_scorer_mode == "candidate_history_concat":
            scorer_input = torch.cat([candidates, retrieved_history], dim=-1)
        else:
            scorer_input = retrieved_history

        eviction_logits = self.scorer(scorer_input).squeeze(-1).unsqueeze(0)
        pred_reuse_dist = self.reuse_distance_estimator(
            scorer_input
        ).squeeze(-1).unsqueeze(0)

        return eviction_logits, pred_reuse_dist

    def initial_history_state(self, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized history state."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        return h, c

    @staticmethod
    def _candidate_subset(
        num_candidates: int,
        oracle_target: int,
        max_candidates: Optional[int],
    ) -> Tuple[List[int], int]:
        if max_candidates is None or num_candidates <= max_candidates:
            return list(range(num_candidates)), oracle_target

        max_candidates = max(2, max_candidates)
        remaining = [idx for idx in range(num_candidates) if idx != oracle_target]
        quota = min(max_candidates - 1, len(remaining))
        if quota <= 0:
            return [oracle_target], 0

        stride = len(remaining) / quota
        chosen = []
        used = {oracle_target}
        for slot in range(quota):
            idx = remaining[min(int(slot * stride), len(remaining) - 1)]
            if idx not in used:
                chosen.append(idx)
                used.add(idx)

        if len(chosen) < quota:
            for idx in remaining:
                if idx not in used:
                    chosen.append(idx)
                    used.add(idx)
                    if len(chosen) >= quota:
                        break

        selected = sorted([oracle_target] + chosen)
        return selected, selected.index(oracle_target)

    def _transform_oracle_distances(
        self,
        oracle_distances,
        selected_indices: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        selected = [float(oracle_distances[idx]) for idx in selected_indices]
        transformed = []
        cap = float(self.reuse_distance_log_cap)
        for distance in selected:
            if not math.isfinite(distance):
                transformed.append(cap)
            elif distance <= 1:
                transformed.append(0.0)
            else:
                transformed.append(min(math.log10(distance), cap))

        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def _ce_target_distribution(
        self,
        oracle_distances,
        selected_indices: List[int],
        target_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.ce_target_policy == "top_set" and oracle_distances is not None:
            relevances = self._transform_oracle_distances(
                oracle_distances,
                selected_indices,
                device,
            )
            max_relevance = relevances.max()
            top_mask = torch.isclose(
                relevances,
                max_relevance.expand_as(relevances),
                rtol=1e-5,
                atol=1e-8,
            )
            top_count = int(top_mask.sum().item())
            if top_count > 0:
                return (top_mask.float() / top_count).unsqueeze(0)

        target = torch.zeros(len(selected_indices), dtype=torch.float32, device=device)
        target[target_idx] = 1.0
        return target.unsqueeze(0)

    @staticmethod
    def _distribution_cross_entropy(
        logits: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_distribution * log_probs).sum(dim=-1).mean()

    def _approx_ndcg_loss(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable NDCG loss.

        Higher eviction scores should receive smaller approximate positions:
        pos_i = 1 + sum_{j != i} sigmoid(alpha * (score_j - score_i)).
        """
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        scores = scores.masked_fill(~mask, -1e9)
        relevances = relevances.masked_fill(~mask, 0.0)

        score_j_minus_i = scores.unsqueeze(1) - scores.unsqueeze(2)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(scores.size(1), dtype=torch.bool, device=scores.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        pairwise = torch.sigmoid(self.ndcg_alpha * score_j_minus_i) * pair_mask.float()
        positions = 1.0 + pairwise.sum(dim=-1)

        gains = torch.expm1(relevances) * mask.float()
        dcg = gains / torch.log2(positions + 1.0)

        sorted_gains = torch.sort(gains, dim=-1, descending=True).values
        ideal_positions = torch.arange(
            1,
            scores.size(1) + 1,
            device=scores.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        idcg = sorted_gains / torch.log2(ideal_positions + 1.0)
        return -(dcg.sum(dim=-1) / (idcg.sum(dim=-1) + 1e-8))

    def loss(
        self,
        snapshots,
        max_candidates: Optional[int] = None,
        max_steps_per_snapshot: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss from snapshots collected by TrieTrainingCache.

        Mirrors original Parrot: loss() calls self() (forward) directly.
        forward(inference=False) re-encodes candidate paths with gradient.
        """
        device = next(self.parameters()).device
        ranking_losses = []
        reuse_losses = []
        ce_losses = []
        stats = {
            "full_steps": 0,
            "capped_steps": 0,
            "candidate_count": 0,
        }

        for snapshot in snapshots:
            eviction_steps = snapshot.eviction_steps
            if (
                max_steps_per_snapshot is not None
                and len(eviction_steps) > max_steps_per_snapshot
            ):
                quota = max(1, max_steps_per_snapshot)
                stride = len(eviction_steps) / quota
                eviction_steps = [
                    eviction_steps[min(int(slot * stride), len(eviction_steps) - 1)]
                    for slot in range(quota)
                ]

            for step in eviction_steps:
                if step.num_candidates < 2:
                    continue

                history_paths = getattr(step, "history_paths", None)
                if history_paths is not None:
                    history_memory = self._encode_history_paths(history_paths, device)
                else:
                    if hasattr(step, "history_tokens"):
                        raise ValueError(
                            "Trie-PARROT v1 snapshots must provide history_paths; "
                            "history_tokens is a legacy prefix/token-history format"
                        )
                    history_memory = None

                selected_indices, target_idx = self._candidate_subset(
                    step.num_candidates,
                    step.oracle_target,
                    max_candidates,
                )
                if len(selected_indices) == step.num_candidates:
                    stats["full_steps"] += 1
                else:
                    stats["capped_steps"] += 1
                stats["candidate_count"] += len(selected_indices)

                candidate_paths = [step.leaf_paths[idx] for idx in selected_indices]
                logits, pred_log_reuse = self(
                    history_memory,
                    candidate_paths=candidate_paths,
                    inference=False,
                )

                oracle_distances = getattr(step, "oracle_distances", None)
                if oracle_distances is not None:
                    relevances = self._transform_oracle_distances(
                        oracle_distances,
                        selected_indices,
                        device,
                    ).unsqueeze(0)
                    ranking_losses.append(self._approx_ndcg_loss(logits, relevances))

                    if self.reuse_loss_weight > 0:
                        reuse_losses.append(F.mse_loss(pred_log_reuse, relevances))

                if self.ce_loss_weight > 0:
                    target_distribution = self._ce_target_distribution(
                        oracle_distances,
                        selected_indices,
                        target_idx,
                        device,
                    )
                    ce_losses.append(
                        self._distribution_cross_entropy(logits, target_distribution)
                    )

        losses = {}
        if ranking_losses:
            losses["ranking"] = (
                self.ranking_loss_weight * torch.cat(ranking_losses, dim=0).mean()
            )
        else:
            losses["ranking"] = torch.tensor(0.0, device=device, requires_grad=True)

        if reuse_losses:
            losses["reuse"] = self.reuse_loss_weight * torch.stack(reuse_losses).mean()
        else:
            losses["reuse"] = torch.tensor(0.0, device=device, requires_grad=True)

        if ce_losses:
            losses["ce"] = self.ce_loss_weight * torch.stack(ce_losses).mean()
        else:
            losses["ce"] = torch.tensor(0.0, device=device, requires_grad=True)

        self.last_loss_stats = stats
        return losses

    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: Optional[str] = None) -> "TrieParrotModel":
        """
        Create model from config file.

        Args:
            config_path: Path to JSON config file
            checkpoint_path: Optional path to model checkpoint

        Returns:
            Initialized TrieParrotModel
        """
        import json
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(
            vocab_size=config["vocab_size"],
            node_embed_dim=config.get("node_embed_dim", 64),
            history_embed_dim=config.get("history_embed_dim", 64),
            hidden_size=config.get("hidden_size", 128),
            max_attention_history=config.get("max_attention_history", 30),
            ranking_loss_weight=config.get("ranking_loss_weight", 1.0),
            reuse_loss_weight=config.get("reuse_loss_weight", 0.1),
            ce_loss_weight=config.get("ce_loss_weight", 0.0),
            ce_target_policy=config.get("ce_target_policy", "argmax"),
            candidate_scorer_mode=config.get("candidate_scorer_mode", "history_only"),
            reuse_distance_log_cap=config.get("reuse_distance_log_cap", 5.0),
            ndcg_alpha=config.get("ndcg_alpha", 10.0),
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)

        return model
