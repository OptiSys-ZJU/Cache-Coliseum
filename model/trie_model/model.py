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


class _ForwardPathEncodingCache:
    """Forward-local differentiable prefix-state cache for trie path encoding."""

    def __init__(self, model: "TrieParrotModel", device: torch.device):
        self.model = model
        self.device = device
        self.states: Dict[Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

    def encode_paths(self, paths) -> torch.Tensor:
        path_list = [tuple(path) for path in paths]
        batch_size = len(path_list)
        if batch_size == 0:
            return torch.zeros(0, self.model.hidden_size, device=self.device)

        missing_by_depth: Dict[int, List[Tuple[int, ...]]] = {}
        missing_seen = set()
        for path in path_list:
            for depth in range(1, len(path) + 1):
                prefix = path[:depth]
                if prefix in self.states or prefix in missing_seen:
                    continue
                missing_seen.add(prefix)
                missing_by_depth.setdefault(depth, []).append(prefix)

        for depth in sorted(missing_by_depth):
            prefixes = missing_by_depth[depth]
            node_ids = torch.tensor(
                [prefix[-1] for prefix in prefixes],
                dtype=torch.long,
                device=self.device,
            )
            node_embeds = self.model.node_embedder(node_ids)
            if depth == 1:
                parent_state = None
            else:
                parent_h = torch.cat(
                    [self.states[prefix[:-1]][0] for prefix in prefixes],
                    dim=0,
                )
                parent_c = torch.cat(
                    [self.states[prefix[:-1]][1] for prefix in prefixes],
                    dim=0,
                )
                parent_state = (parent_h, parent_c)

            h, c = self.model.path_lstm(node_embeds, parent_state)
            for row_idx, prefix in enumerate(prefixes):
                self.states[prefix] = (
                    h[row_idx:row_idx + 1],
                    c[row_idx:row_idx + 1],
                )

        encoded_rows = []
        for path in path_list:
            if path:
                encoded_rows.append(self.states[path][0])
            else:
                encoded_rows.append(
                    torch.zeros(1, self.model.hidden_size, device=self.device)
                )
        return torch.cat(encoded_rows, dim=0)


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
        hidden_size: int = 128,
        max_attention_history: int = 30,
        ranking_loss_weight: float = 1.0,
        reuse_loss_weight: float = 0.1,
        ce_loss_weight: float = 0.0,
        ce_target_policy: str = "argmax",
        top_set_ce_weight: float = 0.0,
        hard_lru_margin_weight: float = 0.0,
        hard_lru_margin: float = 0.2,
        train_on_eviction_decision: bool = False,
        eviction_decision_loss_weight: float = 1.0,
        microstep_access_loss_weight: float = 1.0,
        max_request_history: Optional[int] = None,
        max_microstep_history: Optional[int] = None,
        lru_feature_dim: int = 5,
        reuse_distance_log_cap: float = 5.0,
        ndcg_alpha: float = 10.0,
        lru_prior_alpha_init: float = 0.75,
        lru_prior_alpha_min: float = 0.0,
        lru_prior_alpha_max: float = 1.5,
        lru_prior_alpha_learnable: bool = True,
        use_lcp_features: bool = False,
        lcp_wrong_margin_weight: float = 0.0,
        lcp_wrong_margin: float = 0.2,
        lcp_wrong_ratio_threshold: float = 0.5,
    ):
        """
        Initialize TrieParrotModel.

        Args:
            vocab_size: Size of node ID vocabulary
            node_embed_dim: Dimension of node embeddings (for tree paths)
            hidden_size: Dimension of LSTM hidden states (both history and tree)
            max_attention_history: Max number of past history states for attention
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_attention_history = max_attention_history
        self.max_request_history = (
            max_attention_history
            if max_request_history is None
            else max_request_history
        )
        self.max_microstep_history = (
            max_attention_history
            if max_microstep_history is None
            else max_microstep_history
        )
        self.lru_feature_dim = lru_feature_dim
        requested_lru_prior_alpha_init = float(lru_prior_alpha_init)
        self.lru_prior_alpha_min = float(lru_prior_alpha_min)
        self.lru_prior_alpha_max = float(lru_prior_alpha_max)
        if self.lru_prior_alpha_min < 0:
            raise ValueError("lru_prior_alpha_min must be nonnegative")
        if self.lru_prior_alpha_max <= 0:
            raise ValueError("lru_prior_alpha_max must be positive")
        if self.lru_prior_alpha_max < self.lru_prior_alpha_min:
            raise ValueError(
                "lru_prior_alpha_max must be greater than or equal to "
                "lru_prior_alpha_min"
            )
        self.lru_prior_alpha_range = (
            self.lru_prior_alpha_max - self.lru_prior_alpha_min
        )
        self.lru_prior_alpha_init = min(
            max(requested_lru_prior_alpha_init, self.lru_prior_alpha_min),
            self.lru_prior_alpha_max,
        )
        self.lru_prior_alpha_learnable = bool(lru_prior_alpha_learnable)
        self.ranking_loss_weight = ranking_loss_weight
        self.reuse_loss_weight = reuse_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.top_set_ce_weight = float(top_set_ce_weight)
        self.hard_lru_margin_weight = float(hard_lru_margin_weight)
        self.hard_lru_margin = float(hard_lru_margin)
        self.use_lcp_features = bool(use_lcp_features)
        self.lcp_feature_dim = len(self._lcp_stat_fields())
        self.lcp_wrong_margin_weight = float(lcp_wrong_margin_weight)
        self.lcp_wrong_margin = float(lcp_wrong_margin)
        self.lcp_wrong_ratio_threshold = float(lcp_wrong_ratio_threshold)
        self.train_on_eviction_decision = bool(train_on_eviction_decision)
        self.eviction_decision_loss_weight = float(eviction_decision_loss_weight)
        self.microstep_access_loss_weight = float(microstep_access_loss_weight)
        if ce_target_policy not in {"argmax", "top_set"}:
            raise ValueError(
                "ce_target_policy must be one of {'argmax', 'top_set'}, "
                f"got {ce_target_policy!r}"
            )
        self.ce_target_policy = ce_target_policy
        self.reuse_distance_log_cap = reuse_distance_log_cap
        self.ndcg_alpha = ndcg_alpha
        self.last_loss_stats = {
            "full_steps": 0,
            "capped_steps": 0,
            "candidate_count": 0,
        }

        # --- Node embedding for tree paths and path-history slots ---
        self.node_embedder = NodeEmbedder(vocab_size, node_embed_dim)

        # --- Path Encoder: Tree-LSTM for encoding root-to-leaf paths ---
        self.path_lstm = PathLSTMCell(node_embed_dim, hidden_size)

        # --- Attention: candidate queries attend over ordered path memory ---
        pos_history_size = max(
            1,
            max_attention_history,
            self.max_request_history,
            self.max_microstep_history,
        )
        self.history_pos_embed = nn.Embedding(pos_history_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        self.request_head = nn.Linear(hidden_size, 1)
        self.micro_head = nn.Linear(hidden_size, 1)
        if self.lru_prior_alpha_learnable:
            if self.lru_prior_alpha_range > 0:
                alpha_position = (
                    (self.lru_prior_alpha_init - self.lru_prior_alpha_min)
                    / self.lru_prior_alpha_range
                )
                raw_alpha = self._inverse_sigmoid(alpha_position)
            else:
                raw_alpha = 0.0
            self.lru_prior_raw_alpha = nn.Parameter(torch.tensor(raw_alpha))
        else:
            self.register_buffer(
                "lru_prior_fixed_alpha",
                torch.tensor(self.lru_prior_alpha_init, dtype=torch.float32),
            )
        self.register_buffer(
            "lru_prior_alpha_encoding_version",
            torch.tensor(2, dtype=torch.long),
        )
        self.score_mix_logits = nn.Parameter(torch.zeros(2))
        if self.use_lcp_features:
            lcp_hidden = max(4, min(hidden_size, 16))
            self.lcp_head = nn.Sequential(
                nn.Linear(self.lcp_feature_dim, lcp_hidden),
                nn.ReLU(),
                nn.Linear(lcp_hidden, 1),
            )
        self.reuse_estimator = nn.Linear(
            hidden_size * 2 + lru_feature_dim,
            1,
        )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0.0:
            return -20.0
        return math.log(math.expm1(value))

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        value = min(max(float(value), 1e-7), 1.0 - 1e-7)
        return math.log(value / (1.0 - value))

    @staticmethod
    def _inverse_sigmoid_tensor(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(value / (1.0 - value))

    def lru_prior_alpha(self) -> torch.Tensor:
        if self.lru_prior_alpha_learnable:
            return self.lru_prior_alpha_min + self.lru_prior_alpha_range * (
                torch.sigmoid(self.lru_prior_raw_alpha)
            )
        return torch.clamp(
            self.lru_prior_fixed_alpha,
            min=self.lru_prior_alpha_min,
            max=self.lru_prior_alpha_max,
        )

    def _adapt_state_dict_for_lru_prior(self, state_dict):
        adapted = dict(state_dict)
        dropped_keys = []
        migrated = False

        for key in list(adapted):
            if key.startswith("lru_head."):
                dropped_keys.append(key)
                del adapted[key]
                migrated = True
            elif key.startswith("lcp_head.") and not self.use_lcp_features:
                dropped_keys.append(key)
                del adapted[key]
                migrated = True

        mix_key = "score_mix_logits"
        if mix_key in adapted:
            saved_mix = adapted[mix_key]
            target_mix = self.score_mix_logits
            if tuple(saved_mix.shape) != tuple(target_mix.shape):
                flat_saved_mix = saved_mix.reshape(-1)
                if flat_saved_mix.numel() == 3 and target_mix.numel() == 2:
                    adapted[mix_key] = flat_saved_mix[:2].clone().view_as(target_mix)
                    migrated = True
                else:
                    dropped_keys.append(mix_key)
                    del adapted[mix_key]
                    migrated = True

        version_key = "lru_prior_alpha_encoding_version"
        raw_key = "lru_prior_raw_alpha"
        if (
            raw_key in adapted
            and version_key not in adapted
            and self.lru_prior_alpha_learnable
            and self.lru_prior_alpha_range > 0
        ):
            saved_raw = adapted[raw_key].detach().float()
            old_alpha = self.lru_prior_alpha_max * torch.sigmoid(saved_raw)
            new_position = (
                (old_alpha - self.lru_prior_alpha_min)
                / self.lru_prior_alpha_range
            )
            adapted[raw_key] = self._inverse_sigmoid_tensor(new_position).to(
                dtype=adapted[raw_key].dtype,
                device=adapted[raw_key].device,
            )
            migrated = True
        adapted.setdefault(version_key, self.lru_prior_alpha_encoding_version)

        return adapted, migrated, dropped_keys

    def load_state_dict_compatible(self, state_dict):
        """Load current or pre-LRU-prior TrieParrot weights with narrow migration."""
        adapted, migrated, dropped_keys = self._adapt_state_dict_for_lru_prior(
            state_dict
        )
        incompatible = super().load_state_dict(adapted, strict=False)

        allowed_missing = set()
        current_state_keys = set(self.state_dict())
        for key in ("lru_prior_raw_alpha", "lru_prior_fixed_alpha"):
            if key in current_state_keys and key not in adapted:
                allowed_missing.add(key)
        if self.use_lcp_features:
            for key in current_state_keys:
                if key.startswith("lcp_head.") and key not in adapted:
                    allowed_missing.add(key)
        if "score_mix_logits" in dropped_keys:
            allowed_missing.add("score_mix_logits")

        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        disallowed_missing = missing - allowed_missing
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Unexpected TrieParrot checkpoint mismatch after LRU-prior "
                f"migration: missing={sorted(disallowed_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        return {
            "migrated": migrated or bool(missing & allowed_missing),
            "dropped_keys": dropped_keys,
            "missing_keys": sorted(missing),
        }

    def _model_device(self) -> torch.device:
        """Return this module's device, including DataParallel replicas."""
        for param in self.parameters():
            return param.device

        for module in self.modules():
            former_parameters = getattr(module, "_former_parameters", None)
            if former_parameters:
                for param in former_parameters.values():
                    return param.device

        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def _new_path_encoding_cache(
        self,
        device: Optional[torch.device] = None,
    ) -> _ForwardPathEncodingCache:
        if device is None:
            device = self._model_device()
        return _ForwardPathEncodingCache(self, device)

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
        device = self._model_device()
        node_embed = self.node_embedder.embed_single(node_id, device)
        h, c = self.path_lstm(node_embed, parent_state)
        return h, c

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
        if isinstance(history_memory, list):
            if len(history_memory) == 0:
                return torch.zeros(1, self.hidden_size, device=device), False
            history_memory = torch.cat(history_memory, dim=0)
        if history_memory.dim() == 1:
            history_memory = history_memory.unsqueeze(0)
        return history_memory.to(device), history_memory.numel() > 0

    def _add_history_positions(
        self,
        history_memory: torch.Tensor,
        max_history: Optional[int] = None,
    ) -> torch.Tensor:
        """Add oldest-to-newest positional embeddings to recent history slots."""
        limit = max(1, max_history or self.max_microstep_history)
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
        max_history: Optional[int] = None,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> Optional[torch.Tensor]:
        if history_paths is None:
            return None

        paths = [tuple(path) for path in history_paths]
        if not paths:
            return None

        limit = max(1, max_history or self.max_microstep_history)
        return self._encode_path_batch(
            paths[-limit:],
            device,
            deduplicate=True,
            cache=cache,
        )

    def _encode_request_history_paths(
        self,
        history_paths,
        device: torch.device,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> Optional[torch.Tensor]:
        return self._encode_history_paths(
            history_paths,
            device,
            max_history=self.max_request_history,
            cache=cache,
        )

    def _encode_path_batch(
        self,
        paths,
        device: torch.device,
        deduplicate: bool = False,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> torch.Tensor:
        """Encode many variable-length root-to-node paths in one LSTM pass."""
        path_list = [tuple(path) for path in paths]
        batch_size = len(path_list)
        if batch_size == 0:
            return torch.zeros(0, self.hidden_size, device=device)

        if cache is not None:
            if cache.device != device:
                raise ValueError("path encoding cache device does not match encode device")
            return cache.encode_paths(path_list)

        if deduplicate and batch_size > 1:
            unique_paths = []
            inverse_indices = []
            seen = {}
            for path in path_list:
                unique_idx = seen.get(path)
                if unique_idx is None:
                    unique_idx = len(unique_paths)
                    seen[path] = unique_idx
                    unique_paths.append(path)
                inverse_indices.append(unique_idx)

            if len(unique_paths) < batch_size:
                unique_encoded = self._encode_path_batch(
                    unique_paths,
                    device,
                    deduplicate=False,
                )
                inverse = torch.tensor(
                    inverse_indices,
                    dtype=torch.long,
                    device=device,
                )
                return unique_encoded.index_select(0, inverse)

        lengths = torch.tensor(
            [len(path) for path in path_list],
            dtype=torch.long,
            device=device,
        )
        max_len = int(lengths.max().item()) if batch_size else 0
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        if max_len == 0:
            return h

        path_ids = torch.tensor(
            [
                list(path) + [0] * (max_len - len(path))
                for path in path_list
            ],
            dtype=torch.long,
            device=device,
        )

        embeddings = self.node_embedder(path_ids)
        for step_idx in range(max_len):
            h_new, c_new = self.path_lstm(embeddings[:, step_idx, :], (h, c))
            active = (lengths > step_idx).unsqueeze(1)
            h = torch.where(active, h_new, h)
            c = torch.where(active, c_new, c)
        return h

    def _encode_history_paths_batch(
        self,
        history_paths_batch,
        device: torch.device,
        max_history: Optional[int] = None,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded history memory and a valid-slot mask for a step batch."""
        limit = max(1, max_history or self.max_microstep_history)
        prepared = []
        flat_paths = []
        max_history = 1
        for history_paths in history_paths_batch:
            if history_paths is None:
                paths = []
            else:
                paths = [tuple(path) for path in history_paths][-limit:]
            prepared.append(paths)
            flat_paths.extend(paths)
            max_history = max(max_history, len(paths) if paths else 1)

        encoded = self._encode_path_batch(
            flat_paths,
            device,
            deduplicate=True,
            cache=cache,
        )
        memory = torch.zeros(
            len(prepared),
            max_history,
            self.hidden_size,
            device=device,
        )
        mask = torch.zeros(
            len(prepared),
            max_history,
            dtype=torch.bool,
            device=device,
        )

        offset = 0
        for batch_idx, paths in enumerate(prepared):
            if not paths:
                mask[batch_idx, 0] = True
                continue

            count = len(paths)
            memory[batch_idx, :count, :] = encoded[offset:offset + count]
            positions = torch.arange(count, dtype=torch.long, device=device)
            memory[batch_idx, :count, :] = (
                memory[batch_idx, :count, :]
                + self.history_pos_embed(positions)
            )
            mask[batch_idx, :count] = True
            offset += count

        return memory, mask

    def _encode_request_history_paths_batch(
        self,
        history_paths_batch,
        device: torch.device,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._encode_history_paths_batch(
            history_paths_batch,
            device,
            max_history=self.max_request_history,
            cache=cache,
        )

    def _encode_candidate_paths_batch(
        self,
        candidate_paths_batch,
        device: torch.device,
        cache: Optional[_ForwardPathEncodingCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded candidate states and a valid-candidate mask."""
        counts = [len(candidate_paths) for candidate_paths in candidate_paths_batch]
        max_candidates = max(counts) if counts else 0
        flat_paths = [
            tuple(path)
            for candidate_paths in candidate_paths_batch
            for path in candidate_paths
        ]
        encoded = self._encode_path_batch(flat_paths, device, cache=cache)
        candidates = torch.zeros(
            len(candidate_paths_batch),
            max_candidates,
            self.hidden_size,
            device=device,
        )
        mask = torch.zeros(
            len(candidate_paths_batch),
            max_candidates,
            dtype=torch.bool,
            device=device,
        )

        offset = 0
        for batch_idx, count in enumerate(counts):
            if count == 0:
                continue
            candidates[batch_idx, :count, :] = encoded[offset:offset + count]
            mask[batch_idx, :count] = True
            offset += count
        return candidates, mask

    def _prepare_lru_features_batch(
        self,
        lru_features_batch,
        candidate_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return padded, log-scaled LRU feature tensor of shape (B, N, F)."""
        if lru_features_batch is None:
            raise ValueError("lru_features_batch is required")
        batch_size, max_candidates = candidate_mask.shape
        features = torch.zeros(
            batch_size,
            max_candidates,
            self.lru_feature_dim,
            dtype=torch.float32,
            device=device,
        )
        for batch_idx, row_features in enumerate(lru_features_batch):
            if row_features is None:
                raise ValueError("lru_features are required for every training step")
            row = torch.tensor(
                row_features,
                dtype=torch.float32,
                device=device,
            )
            if row.dim() == 1:
                row = row.unsqueeze(0)
            expected = int(candidate_mask[batch_idx].sum().item())
            if row.size(0) < expected:
                raise ValueError(
                    "lru_features must have one row per selected candidate"
                )
            if row.size(1) != self.lru_feature_dim:
                raise ValueError(
                    "lru_features width must match model.lru_feature_dim"
                )
            if expected > 0:
                features[batch_idx, :expected, :] = row[:expected, :]

        return self._scale_lru_features(features)

    def _prepare_lru_features(
        self,
        lru_features,
        num_candidates: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lru_features is None and num_candidates > 0:
            raise ValueError("lru_features are required for runtime scoring")
        features = torch.zeros(
            1,
            num_candidates,
            self.lru_feature_dim,
            dtype=torch.float32,
            device=device,
        )
        if num_candidates > 0:
            row = torch.tensor(lru_features, dtype=torch.float32, device=device)
            if row.dim() == 1:
                row = row.unsqueeze(0)
            if row.size(0) < num_candidates:
                raise ValueError("lru_features must have one row per candidate")
            if row.size(1) != self.lru_feature_dim:
                raise ValueError(
                    "lru_features width must match model.lru_feature_dim"
                )
            if num_candidates > 0:
                features[0, :num_candidates, :] = row[:num_candidates, :]
        return self._scale_lru_features(features)

    def _scale_lru_features(self, features: torch.Tensor) -> torch.Tensor:
        """Log-scale recency ages while leaving depth-style fields readable."""
        if features.size(-1) == 0:
            return features
        scaled = features.clone()
        age_width = min(4, scaled.size(-1))
        scaled[..., :age_width] = torch.log1p(torch.clamp_min(
            scaled[..., :age_width],
            0.0,
        ))
        if scaled.size(-1) > 4:
            scaled[..., 4:] = torch.log1p(torch.clamp_min(
                scaled[..., 4:],
                0.0,
            ))
        return scaled

    def _scale_lcp_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale LCP feature rows in the same order as _lcp_stat_fields()."""
        if features.size(-1) == 0:
            return features
        scaled = features.clone()
        scaled[..., 0] = torch.log1p(torch.clamp_min(scaled[..., 0], 0.0))
        scaled[..., 1] = torch.clamp(scaled[..., 1], 0.0, 1.0)
        scaled[..., 2] = torch.clamp(scaled[..., 2], 0.0, 1.0)
        scaled[..., 3] = torch.log1p(torch.clamp_min(scaled[..., 3], 0.0))
        scaled[..., 4] = torch.log1p(torch.clamp_min(scaled[..., 4], 0.0))
        return scaled

    def _diagnostics_to_lcp_tensor(self, diagnostics, device: torch.device):
        if diagnostics is None:
            return None
        if isinstance(diagnostics, torch.Tensor):
            row = diagnostics.to(device=device, dtype=torch.float32)
            if row.dim() == 1:
                row = row.unsqueeze(0)
            return row
        rows = [
            self._lcp_feature_row_from_diagnostic(diagnostic)
            for diagnostic in diagnostics
        ]
        if not rows:
            return torch.zeros(
                0,
                self.lcp_feature_dim,
                dtype=torch.float32,
                device=device,
            )
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _prepare_lcp_features_batch(
        self,
        lcp_features_batch,
        candidate_mask: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return padded, scaled LCP feature tensor of shape (B, N, 5)."""
        if not self.use_lcp_features:
            return None
        batch_size, max_candidates = candidate_mask.shape
        features = torch.zeros(
            batch_size,
            max_candidates,
            self.lcp_feature_dim,
            dtype=torch.float32,
            device=device,
        )
        if lcp_features_batch is None:
            return self._scale_lcp_features(features)

        for batch_idx, row_features in enumerate(lcp_features_batch):
            if row_features is None:
                continue
            row = self._diagnostics_to_lcp_tensor(row_features, device)
            expected = int(candidate_mask[batch_idx].sum().item())
            if row is None:
                continue
            if row.size(0) < expected:
                raise ValueError(
                    "lcp_features must have one row per selected candidate"
                )
            if row.size(1) != self.lcp_feature_dim:
                raise ValueError(
                    "lcp_features width must match model.lcp_feature_dim"
                )
            if expected > 0:
                features[batch_idx, :expected, :] = row[:expected, :]

        return self._scale_lcp_features(features)

    def _prepare_lcp_features(
        self,
        lcp_features,
        num_candidates: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.use_lcp_features:
            return None
        features = torch.zeros(
            1,
            num_candidates,
            self.lcp_feature_dim,
            dtype=torch.float32,
            device=device,
        )
        if lcp_features is not None and num_candidates > 0:
            row = self._diagnostics_to_lcp_tensor(lcp_features, device)
            if row.size(0) < num_candidates:
                raise ValueError("lcp_features must have one row per candidate")
            if row.size(1) != self.lcp_feature_dim:
                raise ValueError(
                    "lcp_features width must match model.lcp_feature_dim"
                )
            features[0, :num_candidates, :] = row[:num_candidates, :]
        return self._scale_lcp_features(features)

    def _attend_encoded_history(
        self,
        candidate_states: torch.Tensor,
        history_memory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query_proj(candidate_states)
        memory_keys = self.key_proj(history_memory)

        scale = self.hidden_size ** 0.5
        attn_logits = torch.bmm(queries, memory_keys.transpose(1, 2)) / scale
        attn_logits = attn_logits.masked_fill(~history_mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, history_memory)

    def _combine_score_heads(
        self,
        request_context: torch.Tensor,
        micro_context: torch.Tensor,
        lru_features: torch.Tensor,
        lcp_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        request_logit = self.request_head(request_context).squeeze(-1)
        micro_logit = self.micro_head(micro_context).squeeze(-1)
        lru_prior = lru_features[..., 0]

        mix_weights = F.softmax(self.score_mix_logits, dim=0)
        context_score = (
            mix_weights[0] * request_logit
            + mix_weights[1] * micro_logit
        )
        if self.use_lcp_features:
            if lcp_features is None:
                lcp_features = torch.zeros(
                    *lru_features.shape[:-1],
                    self.lcp_feature_dim,
                    dtype=lru_features.dtype,
                    device=lru_features.device,
                )
            context_score = context_score + self.lcp_head(lcp_features).squeeze(-1)
        eviction_logits = context_score + self.lru_prior_alpha() * lru_prior

        reuse_input = torch.cat(
            [request_context, micro_context, lru_features],
            dim=-1,
        )
        pred_reuse_dist = self.reuse_estimator(
            reuse_input,
        ).squeeze(-1)
        return eviction_logits, pred_reuse_dist

    def _forward_batched_encoded(
        self,
        micro_history_memory: torch.Tensor,
        micro_history_mask: torch.Tensor,
        request_history_memory: torch.Tensor,
        request_history_mask: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
        lru_features: torch.Tensor,
        lcp_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a padded batch of training steps."""
        micro_context = self._attend_encoded_history(
            candidate_states,
            micro_history_memory,
            micro_history_mask,
        )
        request_context = self._attend_encoded_history(
            candidate_states,
            request_history_memory,
            request_history_mask,
        )
        eviction_logits, pred_reuse_dist = self._combine_score_heads(
            request_context,
            micro_context,
            lru_features,
            lcp_features,
        )

        eviction_logits = eviction_logits.masked_fill(~candidate_mask, -1e9)
        pred_reuse_dist = pred_reuse_dist.masked_fill(~candidate_mask, 0.0)
        return eviction_logits, pred_reuse_dist

    def forward_batched(
        self,
        microstep_history_paths_batch,
        candidate_paths_batch,
        request_history_paths_batch,
        lru_features_batch,
        lcp_features_batch=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training scores for a batch of microsteps.

        Candidate paths are encoded as attention queries. They retrieve one
        context from request-level leaf history and one context from current
        request microstep history; explicit candidate LRU features provide the
        third score head.
        """
        device = self._model_device()
        active_cache = getattr(self, "_active_path_encoding_cache", None)
        path_cache = active_cache or self._new_path_encoding_cache(device)
        micro_memory, micro_mask = self._encode_history_paths_batch(
            microstep_history_paths_batch,
            device,
            max_history=self.max_microstep_history,
            cache=path_cache,
        )
        request_memory, request_mask = self._encode_request_history_paths_batch(
            request_history_paths_batch,
            device,
            cache=path_cache,
        )
        candidate_states, candidate_mask = self._encode_candidate_paths_batch(
            candidate_paths_batch,
            device,
            cache=path_cache,
        )
        lru_features = self._prepare_lru_features_batch(
            lru_features_batch,
            candidate_mask,
            device,
        )
        lcp_features = self._prepare_lcp_features_batch(
            lcp_features_batch,
            candidate_mask,
            device,
        )
        logits, pred_reuse = self._forward_batched_encoded(
            micro_memory,
            micro_mask,
            request_memory,
            request_mask,
            candidate_states,
            candidate_mask,
            lru_features,
            lcp_features,
        )
        return logits, pred_reuse, candidate_mask

    def forward(
        self,
        microstep_history_memory: torch.Tensor,
        request_history_memory: torch.Tensor,
        lru_features,
        candidate_states: List[torch.Tensor] = None,
        candidate_paths: List[Tuple[int, ...]] = None,
        lcp_features=None,
        inference: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eviction scores for candidate leaf nodes.

        When inference=True:
            Uses pre-computed candidate_states from trie cache.
        When inference=False:
            Re-encodes candidate_paths from scratch with gradient.

        Args:
            microstep_history_memory: Current-request prefix memory,
                oldest-to-newest, shape (M, hidden_size).
            candidate_states: List of N pre-computed candidate path hidden states.
            candidate_paths: List of N root-to-leaf node ID tuples.
            request_history_memory: Completed-request leaf memory,
                oldest-to-newest, shape (R, hidden_size).
            lru_features: Raw per-candidate LRU feature rows.
            lcp_features: Optional raw per-candidate LCP feature rows.
            inference: If True, use cached candidate_states; otherwise re-encode
                candidate_paths with gradient.

        Returns:
            eviction_logits: shape (1, N)
            pred_reuse_distances: shape (1, N)
        """
        device = self._model_device()
        micro_memory, has_micro_history = self._prepare_history_memory(
            microstep_history_memory,
            device,
        )
        if has_micro_history:
            micro_memory = self._add_history_positions(
                micro_memory,
                self.max_microstep_history,
            )
        micro_mask = torch.ones(
            1,
            micro_memory.size(0),
            dtype=torch.bool,
            device=device,
        )

        request_memory, has_request_history = self._prepare_history_memory(
            request_history_memory,
            device,
        )
        if has_request_history:
            request_memory = self._add_history_positions(
                request_memory,
                self.max_request_history,
            )
        request_mask = torch.ones(
            1,
            request_memory.size(0),
            dtype=torch.bool,
            device=device,
        )

        if inference:
            assert candidate_states is not None, "candidate_states required when inference=True"
            if len(candidate_states) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            candidates = torch.cat(candidate_states, dim=0)
        else:
            assert candidate_paths is not None, "candidate_paths required when inference=False"
            if len(candidate_paths) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            path_cache = self._new_path_encoding_cache(device)
            candidates = self._encode_path_batch(
                candidate_paths,
                device,
                cache=path_cache,
            )

        candidate_batch = candidates.unsqueeze(0)
        candidate_mask = torch.ones(
            1,
            candidates.size(0),
            dtype=torch.bool,
            device=device,
        )
        lru_feature_tensor = self._prepare_lru_features(
            lru_features,
            candidates.size(0),
            device,
        )
        lcp_feature_tensor = self._prepare_lcp_features(
            lcp_features,
            candidates.size(0),
            device,
        )
        eviction_logits, pred_reuse_dist = self._forward_batched_encoded(
            micro_memory.unsqueeze(0),
            micro_mask,
            request_memory.unsqueeze(0),
            request_mask,
            candidate_batch,
            candidate_mask,
            lru_feature_tensor,
            lcp_feature_tensor,
        )

        return eviction_logits, pred_reuse_dist

    @staticmethod
    def _candidate_subset(
        num_candidates: int,
        oracle_target: int,
        max_candidates: Optional[int],
        required_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], int]:
        if max_candidates is None or num_candidates <= max_candidates:
            return list(range(num_candidates)), oracle_target

        required = {oracle_target}
        if required_indices is not None:
            required.update(
                idx for idx in required_indices
                if 0 <= idx < num_candidates
            )
        max_candidates = max(2, max_candidates, len(required))
        remaining = [idx for idx in range(num_candidates) if idx not in required]
        quota = min(max_candidates - len(required), len(remaining))
        chosen = []
        used = set(required)
        if quota > 0:
            stride = len(remaining) / quota
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

        selected = sorted(required | set(chosen))
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

    @staticmethod
    def loss_names() -> Tuple[str, ...]:
        return (
            "ranking",
            "reuse",
            "ce",
            "top_set_ce",
            "hard_lru_margin",
            "lcp_wrong_margin",
        )

    @staticmethod
    def _oracle_top_set_from_distances(oracle_distances) -> Tuple[int, ...]:
        if oracle_distances is None or len(oracle_distances) == 0:
            return tuple()
        max_distance = max(float(distance) for distance in oracle_distances)
        return tuple(
            idx for idx, distance in enumerate(oracle_distances)
            if float(distance) == max_distance
        )

    def _step_kind_loss_weight(self, step_kind: str) -> float:
        if step_kind == "eviction_decision":
            if not self.train_on_eviction_decision:
                return 0.0
            return self.eviction_decision_loss_weight
        return self.microstep_access_loss_weight

    @staticmethod
    def _lcp_stat_fields() -> Tuple[str, ...]:
        return (
            "lcp_len",
            "lcp_ratio_candidate",
            "lcp_ratio_current",
            "candidate_suffix_len",
            "current_suffix_len",
        )

    @staticmethod
    def _lcp_len(left_path, right_path) -> int:
        length = 0
        for left, right in zip(left_path, right_path):
            if left != right:
                break
            length += 1
        return length

    @classmethod
    def lcp_features_from_paths(cls, candidate_paths, current_path):
        current = tuple(current_path or ())
        rows = []
        for candidate_path in candidate_paths:
            candidate = tuple(candidate_path or ())
            lcp_len = cls._lcp_len(candidate, current)
            rows.append((
                float(lcp_len),
                lcp_len / len(candidate) if candidate else 0.0,
                lcp_len / len(current) if current else 0.0,
                float(max(0, len(candidate) - lcp_len)),
                float(max(0, len(current) - lcp_len)),
            ))
        return tuple(rows)

    @classmethod
    def _lcp_feature_row_from_diagnostic(cls, diagnostic):
        if diagnostic is None:
            return [0.0] * len(cls._lcp_stat_fields())
        if isinstance(diagnostic, dict):
            return [
                float(diagnostic.get(field, 0.0))
                for field in cls._lcp_stat_fields()
            ]
        row = list(diagnostic)
        if len(row) != len(cls._lcp_stat_fields()):
            raise ValueError(
                "lcp_features width must match TrieParrotModel lcp fields"
            )
        return [float(value) for value in row]

    @classmethod
    def _lcp_ratio_current_from_diagnostic(cls, diagnostic) -> float:
        row = cls._lcp_feature_row_from_diagnostic(diagnostic)
        return min(max(float(row[2]), 0.0), 1.0)

    @classmethod
    def _accumulate_lcp_stats(cls, stats: dict, prefix: str, diagnostic):
        if diagnostic is None:
            return
        row = cls._lcp_feature_row_from_diagnostic(diagnostic)
        for field, value in zip(cls._lcp_stat_fields(), row):
            stats[f"{prefix}_{field}_sum"] += float(value)
        stats[f"{prefix}_count"] += 1

    @classmethod
    def _finalize_lcp_stats(cls, stats: dict, prefix: str):
        count = int(stats.get(f"{prefix}_count", 0))
        for field in cls._lcp_stat_fields():
            sum_key = f"{prefix}_{field}_sum"
            mean_key = f"{prefix}_{field}_mean"
            stats[mean_key] = stats.get(sum_key, 0.0) / count if count else 0.0

    def loss(
        self,
        snapshots,
        max_candidates: Optional[int] = None,
        max_steps_per_snapshot: Optional[int] = None,
        warmup_steps_per_snapshot: int = 0,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss from snapshots collected by TrieTrainingCache.

        Time positions are grouped across windows, then candidates/history are
        padded and scored in one batched attention pass. The surrounding field
        is named eviction_steps; entries can be microstep access states and,
        when enabled, true eviction-decision states.
        """
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be one of {'mean', 'sum'}")

        device = self._model_device()
        ranking_losses = []
        reuse_losses = []
        ce_losses = []
        top_set_ce_losses = []
        hard_lru_margin_losses = []
        lcp_wrong_margin_losses = []
        stats = {
            "full_steps": 0,
            "capped_steps": 0,
            "candidate_count": 0,
            "ranking_count": 0,
            "reuse_count": 0,
            "ce_count": 0,
            "top_set_ce_count": 0,
            "hard_lru_margin_count": 0,
            "lcp_wrong_margin_count": 0,
            "warmup_steps": 0,
            "loss_steps": 0,
            "microstep_access_steps": 0,
            "eviction_decision_steps": 0,
            "lru_target_kept_count": 0,
            "lru_target_steps": 0,
            "oracle_top_set_kept_count": 0,
            "oracle_top_set_steps": 0,
            "hard_lru_cases_count": 0,
            "hard_lru_active_frac": 0.0,
            "lcp_wrong_cases_count": 0,
            "lcp_wrong_high_lcp_count": 0,
            "lcp_wrong_margin_active_frac": 0.0,
            "max_loss_candidates_effective": 0,
            "top_set_acc_correct": 0,
            "top_set_acc_count": 0,
            "top_set_acc": 0.0,
            "regret_sum": 0.0,
            "regret_count": 0,
            "regret": 0.0,
            "lru_prior_alpha": float(self.lru_prior_alpha().detach().item()),
        }
        for prefix in (
            "oracle_target_lcp",
            "lru_target_lcp",
            "model_wrong_target_lcp",
        ):
            stats[f"{prefix}_count"] = 0
            for field in self._lcp_stat_fields():
                stats[f"{prefix}_{field}_sum"] = 0.0
                stats[f"{prefix}_{field}_mean"] = 0.0

        step_windows = []
        for snapshot in snapshots:
            eviction_steps = snapshot.eviction_steps
            warmup_steps = max(0, int(warmup_steps_per_snapshot or 0))
            if warmup_steps >= len(eviction_steps) and len(eviction_steps) > 0:
                raise ValueError(
                    "warmup_steps_per_snapshot must be smaller than the "
                    f"number of eviction steps, got {warmup_steps} for "
                    f"{len(eviction_steps)} steps"
                )
            if warmup_steps > 0:
                stats["warmup_steps"] += min(warmup_steps, len(eviction_steps))
                eviction_steps = eviction_steps[warmup_steps:]

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

            prepared_steps = []
            for step in eviction_steps:
                if step.num_candidates < 2:
                    prepared_steps.append(None)
                    continue

                step_kind = getattr(step, "step_kind", "microstep_access")
                step_loss_weight = self._step_kind_loss_weight(step_kind)
                if step_loss_weight <= 0.0:
                    prepared_steps.append(None)
                    continue
                stats["loss_steps"] += 1
                if step_kind == "eviction_decision":
                    stats["eviction_decision_steps"] += 1
                else:
                    stats["microstep_access_steps"] += 1

                microstep_history_paths = getattr(step, "microstep_history_paths", None)
                if microstep_history_paths is None:
                    raise ValueError(
                        "Trie-PARROT lru-trie snapshots must provide "
                        "microstep_history_paths"
                    )
                request_history_paths = getattr(step, "request_history_paths", None)
                if request_history_paths is None:
                    raise ValueError(
                        "Trie-PARROT lru-trie snapshots must provide "
                        "request_history_paths"
                    )
                raw_lru_features = getattr(step, "lru_features", None)
                if raw_lru_features is None:
                    raise ValueError(
                        "Trie-PARROT lru-trie snapshots must provide "
                        "lru_features"
                    )

                oracle_distances = getattr(step, "oracle_distances", None)
                oracle_top_set = tuple(
                    getattr(step, "oracle_top_set", None) or ()
                )
                if not oracle_top_set:
                    oracle_top_set = self._oracle_top_set_from_distances(
                        oracle_distances
                    )
                lru_target = getattr(step, "lru_target", None)
                required_indices = set(
                    getattr(step, "required_candidate_indices", None) or ()
                )
                required_indices.update(oracle_top_set)
                if lru_target is not None:
                    required_indices.add(lru_target)

                selected_indices, target_idx = self._candidate_subset(
                    step.num_candidates,
                    step.oracle_target,
                    max_candidates,
                    required_indices,
                )
                selected_set = set(selected_indices)
                if len(selected_indices) == step.num_candidates:
                    stats["full_steps"] += 1
                else:
                    stats["capped_steps"] += 1
                stats["candidate_count"] += len(selected_indices)
                stats["max_loss_candidates_effective"] = max(
                    stats["max_loss_candidates_effective"],
                    len(selected_indices),
                )
                if lru_target is not None:
                    stats["lru_target_steps"] += 1
                    if lru_target in selected_set:
                        stats["lru_target_kept_count"] += 1
                if oracle_top_set:
                    stats["oracle_top_set_steps"] += 1
                    if set(oracle_top_set).issubset(selected_set):
                        stats["oracle_top_set_kept_count"] += 1
                hard_lru_case = (
                    lru_target is not None
                    and oracle_top_set
                    and lru_target not in set(oracle_top_set)
                    and lru_target in selected_set
                )
                if hard_lru_case:
                    stats["hard_lru_cases_count"] += 1

                candidate_paths = [step.leaf_paths[idx] for idx in selected_indices]
                lru_features = [
                    raw_lru_features[idx] for idx in selected_indices
                ]
                lcp_features = None
                relevances = None
                if oracle_distances is not None:
                    relevances = self._transform_oracle_distances(
                        oracle_distances,
                        selected_indices,
                        device,
                    )

                target_distribution = None
                if self.ce_loss_weight > 0:
                    target_distribution = self._ce_target_distribution(
                        oracle_distances,
                        selected_indices,
                        target_idx,
                        device,
                    ).squeeze(0)

                selected_top_positions = [
                    pos for pos, idx in enumerate(selected_indices)
                    if idx in oracle_top_set
                ]
                if not selected_top_positions:
                    selected_top_positions = [target_idx]
                lru_target_pos = (
                    selected_indices.index(lru_target)
                    if lru_target in selected_set
                    else None
                )

                lcp_diagnostics = tuple(
                    getattr(step, "lcp_diagnostics", None) or ()
                )
                if lcp_diagnostics:
                    lcp_features = [
                        lcp_diagnostics[idx] for idx in selected_indices
                    ]
                    if 0 <= step.oracle_target < len(lcp_diagnostics):
                        self._accumulate_lcp_stats(
                            stats,
                            "oracle_target_lcp",
                            lcp_diagnostics[step.oracle_target],
                        )
                    if (
                        lru_target is not None
                        and 0 <= lru_target < len(lcp_diagnostics)
                    ):
                        self._accumulate_lcp_stats(
                            stats,
                            "lru_target_lcp",
                            lcp_diagnostics[lru_target],
                        )

                prepared_steps.append({
                    "microstep_history_paths": microstep_history_paths,
                    "request_history_paths": request_history_paths,
                    "candidate_paths": candidate_paths,
                    "lru_features": lru_features,
                    "lcp_features": lcp_features,
                    "relevances": relevances,
                    "target_distribution": target_distribution,
                    "step_weight": float(step_loss_weight),
                    "selected_indices": selected_indices,
                    "oracle_top_set": oracle_top_set,
                    "top_positions": selected_top_positions,
                    "lru_target_pos": lru_target_pos,
                    "hard_lru_case": hard_lru_case,
                    "lcp_diagnostics": lcp_diagnostics,
                })

            step_windows.append(prepared_steps)

        max_window_len = max((len(window) for window in step_windows), default=0)
        had_active_cache = hasattr(self, "_active_path_encoding_cache")
        previous_active_cache = getattr(self, "_active_path_encoding_cache", None)
        self._active_path_encoding_cache = self._new_path_encoding_cache(device)
        try:
            for step_idx in range(max_window_len):
                batch_steps = [
                    window[step_idx]
                    for window in step_windows
                    if step_idx < len(window) and window[step_idx] is not None
                ]
                if not batch_steps:
                    continue

                forward_args = (
                    [item["microstep_history_paths"] for item in batch_steps],
                    [item["candidate_paths"] for item in batch_steps],
                    [item["request_history_paths"] for item in batch_steps],
                    [item["lru_features"] for item in batch_steps],
                )
                if self.use_lcp_features:
                    logits, pred_log_reuse, candidate_mask = self.forward_batched(
                        *forward_args,
                        [item["lcp_features"] for item in batch_steps],
                    )
                else:
                    logits, pred_log_reuse, candidate_mask = self.forward_batched(
                        *forward_args
                    )

                max_batch_candidates = logits.size(1)
                step_weights = torch.tensor(
                    [item["step_weight"] for item in batch_steps],
                    dtype=torch.float32,
                    device=device,
                )
                oracle_rows = [
                    idx for idx, item in enumerate(batch_steps)
                    if item["relevances"] is not None
                ]
                if oracle_rows:
                    relevances = torch.zeros(
                        len(batch_steps),
                        max_batch_candidates,
                        dtype=torch.float32,
                        device=device,
                    )
                    for row_idx in oracle_rows:
                        row_relevance = batch_steps[row_idx]["relevances"]
                        relevances[row_idx, :row_relevance.numel()] = row_relevance

                    oracle_row_tensor = torch.tensor(
                        oracle_rows,
                        dtype=torch.long,
                        device=device,
                    )
                    per_step_ranking = self._approx_ndcg_loss(
                        logits.index_select(0, oracle_row_tensor),
                        relevances.index_select(0, oracle_row_tensor),
                        candidate_mask.index_select(0, oracle_row_tensor),
                    )
                    ranking_losses.append(
                        per_step_ranking
                        * step_weights.index_select(0, oracle_row_tensor)
                    )

                    if self.reuse_loss_weight > 0:
                        squared_error = (pred_log_reuse - relevances).pow(2)
                        valid_error = squared_error * candidate_mask.float()
                        per_step_reuse = (
                            valid_error.sum(dim=-1)
                            / candidate_mask.float().sum(dim=-1).clamp_min(1.0)
                        )
                        reuse_losses.append(
                            per_step_reuse.index_select(0, oracle_row_tensor)
                            * step_weights.index_select(0, oracle_row_tensor)
                        )

                if self.ce_loss_weight > 0:
                    target_distribution = torch.zeros(
                        len(batch_steps),
                        max_batch_candidates,
                        dtype=torch.float32,
                        device=device,
                    )
                    for row_idx, item in enumerate(batch_steps):
                        row_target = item["target_distribution"]
                        if row_target is None:
                            continue
                        target_distribution[row_idx, :row_target.numel()] = row_target

                    log_probs = F.log_softmax(logits, dim=-1)
                    ce_losses.append(
                        -(target_distribution * log_probs).sum(dim=-1)
                        * step_weights
                    )

                if (
                    self.top_set_ce_weight > 0
                    or self.hard_lru_margin_weight > 0
                    or self.lcp_wrong_margin_weight > 0
                ):
                    for row_idx, item in enumerate(batch_steps):
                        row_valid_count = int(candidate_mask[row_idx].sum().item())
                        row_logits = logits[row_idx, :row_valid_count]
                        top_positions = [
                            pos for pos in item["top_positions"]
                            if pos < row_valid_count
                        ]
                        if not top_positions:
                            continue
                        top_position_tensor = torch.tensor(
                            top_positions,
                            dtype=torch.long,
                            device=device,
                        )
                        top_score = torch.logsumexp(
                            row_logits.index_select(0, top_position_tensor),
                            dim=0,
                        )
                        all_score = torch.logsumexp(row_logits, dim=0)
                        weight = item["step_weight"]

                        if self.top_set_ce_weight > 0:
                            top_set_ce_losses.append(
                                (all_score - top_score).unsqueeze(0) * weight
                            )

                        lru_target_pos = item["lru_target_pos"]
                        if (
                            self.hard_lru_margin_weight > 0
                            and item["hard_lru_case"]
                            and lru_target_pos is not None
                            and lru_target_pos < row_valid_count
                        ):
                            hard_lru_margin_losses.append(
                                F.softplus(
                                    row_logits[lru_target_pos]
                                    - top_score
                                    + self.hard_lru_margin
                                ).unsqueeze(0)
                                * weight
                            )

                        if (
                            self.lcp_wrong_margin_weight > 0
                            and item["oracle_top_set"]
                            and item["lcp_diagnostics"]
                        ):
                            pred_pos = int(torch.argmax(row_logits.detach()).item())
                            original_pred_idx = item["selected_indices"][pred_pos]
                            if original_pred_idx not in set(item["oracle_top_set"]):
                                stats["lcp_wrong_cases_count"] += 1
                                if 0 <= original_pred_idx < len(item["lcp_diagnostics"]):
                                    ratio_current = (
                                        self._lcp_ratio_current_from_diagnostic(
                                            item["lcp_diagnostics"][original_pred_idx]
                                        )
                                    )
                                else:
                                    ratio_current = 0.0
                                if ratio_current >= self.lcp_wrong_ratio_threshold:
                                    stats["lcp_wrong_high_lcp_count"] += 1
                                    lcp_wrong_margin_losses.append(
                                        F.softplus(
                                            row_logits[pred_pos]
                                            - top_score
                                            + self.lcp_wrong_margin
                                        ).unsqueeze(0)
                                        * weight
                                    )

                for row_idx, item in enumerate(batch_steps):
                    if item["relevances"] is None:
                        continue
                    row_valid_count = int(candidate_mask[row_idx].sum().item())
                    if row_valid_count <= 0:
                        continue
                    row_scores = logits[row_idx, :row_valid_count]
                    pred_pos = int(torch.argmax(row_scores).detach().item())
                    top_positions = [
                        pos for pos in item["top_positions"]
                        if pos < row_valid_count
                    ]
                    if not top_positions:
                        continue
                    stats["top_set_acc_count"] += 1
                    if pred_pos in top_positions:
                        stats["top_set_acc_correct"] += 1

                    row_relevances = item["relevances"].detach()
                    top_relevance = row_relevances[
                        torch.tensor(top_positions, dtype=torch.long, device=device)
                    ].max()
                    pred_relevance = row_relevances[pred_pos]
                    stats["regret_sum"] += float(
                        torch.clamp_min(top_relevance - pred_relevance, 0.0).item()
                    )
                    stats["regret_count"] += 1

                    original_pred_idx = item["selected_indices"][pred_pos]
                    if (
                        original_pred_idx not in set(item["oracle_top_set"])
                        and item["lcp_diagnostics"]
                        and 0 <= original_pred_idx < len(item["lcp_diagnostics"])
                    ):
                        self._accumulate_lcp_stats(
                            stats,
                            "model_wrong_target_lcp",
                            item["lcp_diagnostics"][original_pred_idx],
                        )
        finally:
            if had_active_cache:
                self._active_path_encoding_cache = previous_active_cache
            else:
                delattr(self, "_active_path_encoding_cache")

        def reduce_loss_terms(terms, weight):
            values = torch.cat(terms, dim=0)
            if reduction == "sum":
                return weight * values.sum()
            return weight * values.mean()

        losses = {}
        if ranking_losses:
            stats["ranking_count"] = sum(term.numel() for term in ranking_losses)
            losses["ranking"] = reduce_loss_terms(
                ranking_losses,
                self.ranking_loss_weight,
            )
        else:
            losses["ranking"] = torch.tensor(0.0, device=device, requires_grad=True)

        if reuse_losses:
            stats["reuse_count"] = sum(term.numel() for term in reuse_losses)
            losses["reuse"] = reduce_loss_terms(
                reuse_losses,
                self.reuse_loss_weight,
            )
        else:
            losses["reuse"] = torch.tensor(0.0, device=device, requires_grad=True)

        if ce_losses:
            stats["ce_count"] = sum(term.numel() for term in ce_losses)
            losses["ce"] = reduce_loss_terms(
                ce_losses,
                self.ce_loss_weight,
            )
        else:
            losses["ce"] = torch.tensor(0.0, device=device, requires_grad=True)

        if top_set_ce_losses:
            stats["top_set_ce_count"] = sum(
                term.numel() for term in top_set_ce_losses
            )
            losses["top_set_ce"] = reduce_loss_terms(
                top_set_ce_losses,
                self.top_set_ce_weight,
            )
        else:
            losses["top_set_ce"] = torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        if hard_lru_margin_losses:
            stats["hard_lru_margin_count"] = sum(
                term.numel() for term in hard_lru_margin_losses
            )
            losses["hard_lru_margin"] = reduce_loss_terms(
                hard_lru_margin_losses,
                self.hard_lru_margin_weight,
            )
        else:
            losses["hard_lru_margin"] = torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        if lcp_wrong_margin_losses:
            stats["lcp_wrong_margin_count"] = sum(
                term.numel() for term in lcp_wrong_margin_losses
            )
            losses["lcp_wrong_margin"] = reduce_loss_terms(
                lcp_wrong_margin_losses,
                self.lcp_wrong_margin_weight,
            )
        else:
            losses["lcp_wrong_margin"] = torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        if stats["loss_steps"] > 0:
            stats["hard_lru_active_frac"] = (
                stats["hard_lru_cases_count"] / stats["loss_steps"]
            )
            stats["lcp_wrong_margin_active_frac"] = (
                stats["lcp_wrong_high_lcp_count"] / stats["loss_steps"]
            )
        if stats["top_set_acc_count"] > 0:
            stats["top_set_acc"] = (
                stats["top_set_acc_correct"] / stats["top_set_acc_count"]
            )
        if stats["regret_count"] > 0:
            stats["regret"] = stats["regret_sum"] / stats["regret_count"]

        for prefix in (
            "oracle_target_lcp",
            "lru_target_lcp",
            "model_wrong_target_lcp",
        ):
            self._finalize_lcp_stats(stats, prefix)

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

        fixed_lru_prior_alpha = config.get("lru_prior_alpha_fixed")
        fixed_lru_prior_requested = (
            fixed_lru_prior_alpha
            if isinstance(fixed_lru_prior_alpha, bool)
            else fixed_lru_prior_alpha is not None
        )
        default_lru_prior_alpha = (
            0.75
            if isinstance(fixed_lru_prior_alpha, bool)
            or fixed_lru_prior_alpha is None
            else fixed_lru_prior_alpha
        )
        lru_prior_alpha_init = config.get(
            "lru_prior_alpha_init",
            default_lru_prior_alpha,
        )
        model = cls(
            vocab_size=config["vocab_size"],
            node_embed_dim=config.get("node_embed_dim", 64),
            hidden_size=config.get("hidden_size", 128),
            max_attention_history=config.get("max_attention_history", 30),
            max_request_history=config.get("max_request_history"),
            max_microstep_history=config.get("max_microstep_history"),
            lru_feature_dim=config.get("lru_feature_dim", 5),
            ranking_loss_weight=config.get("ranking_loss_weight", 1.0),
            reuse_loss_weight=config.get("reuse_loss_weight", 0.1),
            ce_loss_weight=config.get("ce_loss_weight", 0.0),
            ce_target_policy=config.get("ce_target_policy", "argmax"),
            top_set_ce_weight=config.get("top_set_ce_weight", 0.0),
            hard_lru_margin_weight=config.get("hard_lru_margin_weight", 0.0),
            hard_lru_margin=config.get("hard_lru_margin", 0.2),
            train_on_eviction_decision=config.get("train_on_eviction_decision", False),
            eviction_decision_loss_weight=config.get(
                "eviction_decision_loss_weight",
                1.0,
            ),
            microstep_access_loss_weight=config.get(
                "microstep_access_loss_weight",
                1.0,
            ),
            reuse_distance_log_cap=config.get("reuse_distance_log_cap", 5.0),
            ndcg_alpha=config.get("ndcg_alpha", 10.0),
            lru_prior_alpha_init=lru_prior_alpha_init,
            lru_prior_alpha_min=config.get("lru_prior_alpha_min", 0.0),
            lru_prior_alpha_max=config.get("lru_prior_alpha_max", 1.5),
            lru_prior_alpha_learnable=config.get(
                "lru_prior_alpha_learnable",
                not fixed_lru_prior_requested,
            ),
            use_lcp_features=config.get("use_lcp_features", False),
            lcp_wrong_margin_weight=config.get("lcp_wrong_margin_weight", 0.0),
            lcp_wrong_margin=config.get("lcp_wrong_margin", 0.2),
            lcp_wrong_ratio_threshold=config.get(
                "lcp_wrong_ratio_threshold",
                0.5,
            ),
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict_compatible(state_dict)

        return model
