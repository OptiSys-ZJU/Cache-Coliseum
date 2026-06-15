"""
TrieParrotModel: Tree-state aware cache eviction predictor.

Candidate leaf paths are encoded by Tree-LSTM and compared against a recent
history memory built by the history LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # --- Node embedding (shared between tree path & history) ---
        self.node_embedder = NodeEmbedder(vocab_size, node_embed_dim)

        # --- History Encoder: LSTM over access history ---
        self.history_lstm = nn.LSTMCell(history_embed_dim, hidden_size)

        # Project node embedding to history input dim (in case they differ)
        if node_embed_dim != history_embed_dim:
            self.history_proj = nn.Linear(node_embed_dim, history_embed_dim)
        else:
            self.history_proj = nn.Identity()

        # --- Path Encoder: Tree-LSTM for encoding root-to-leaf paths ---
        self.path_lstm = PathLSTMCell(node_embed_dim, hidden_size)

        # --- Attention: candidate queries attend over history memory ---
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        # --- Scorer: candidate + retrieved history ---
        scorer_input_dim = hidden_size * 2
        self.scorer = nn.Linear(scorer_input_dim, 1)

        # --- Reuse distance estimator (currently an auxiliary, untrained head) ---
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
        Process one step of access history through the history LSTM.

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
    ) -> torch.Tensor:
        """Normalize the minimal history-memory interface to shape (M, H)."""
        if history_memory is None:
            return torch.zeros(1, self.hidden_size, device=device)
        if isinstance(history_memory, tuple):
            history_memory = history_memory[0]
        if isinstance(history_memory, list):
            if len(history_memory) == 0:
                return torch.zeros(1, self.hidden_size, device=device)
            history_memory = torch.cat(history_memory, dim=0)
        if history_memory.dim() == 1:
            history_memory = history_memory.unsqueeze(0)
        return history_memory.to(device)

    def _encode_history_tokens(
        self,
        history_tokens,
        device: torch.device,
    ) -> torch.Tensor:
        if history_tokens is None:
            return torch.zeros(1, self.hidden_size, device=device)

        state = None
        memories = []
        for node_id in history_tokens:
            state = self.encode_history_step(node_id, state)
            memories.append(state[0])

        if not memories:
            return torch.zeros(1, self.hidden_size, device=device)

        limit = max(1, self.max_attention_history)
        return torch.cat(memories[-limit:], dim=0).to(device)

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
            history_memory: History memory tensor, shape (M, hidden_size). The
                current flow may pass a single hidden state wrapped as length-1 memory.
            candidate_states: List of N pre-computed candidate hidden states.
            candidate_paths: List of N root-to-leaf node ID tuples.
            inference: If True, use cached candidate_states; otherwise re-encode
                candidate_paths with gradient.

        Returns:
            eviction_logits: shape (1, N)
            pred_reuse_distances: shape (1, N)
        """
        device = next(self.parameters()).device
        history_memory = self._prepare_history_memory(history_memory, device)

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

        scorer_input = torch.cat([candidates, retrieved_history], dim=-1)

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
        eviction_losses = []

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

                history_tokens = getattr(step, "history_tokens", None)
                if history_tokens is not None:
                    history_memory = self._encode_history_tokens(history_tokens, device)
                else:
                    history_memory = torch.zeros(1, self.hidden_size, device=device)

                selected_indices, target_idx = self._candidate_subset(
                    step.num_candidates,
                    step.oracle_target,
                    max_candidates,
                )
                candidate_paths = [step.leaf_paths[idx] for idx in selected_indices]
                logits, _ = self(
                    history_memory,
                    candidate_paths=candidate_paths,
                    inference=False,
                )

                target = torch.tensor([target_idx], device=device)
                eviction_losses.append(F.cross_entropy(logits, target))

        losses = {}
        if eviction_losses:
            losses["eviction"] = torch.stack(eviction_losses).mean()
        else:
            losses["eviction"] = torch.tensor(0.0, device=device, requires_grad=True)
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
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)

        return model
