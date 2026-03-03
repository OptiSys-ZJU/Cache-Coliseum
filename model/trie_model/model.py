"""
TrieParrotModel: Tree-state aware cache eviction predictor.

Architecture (PRD 2.2):
  1. History Encoder (LSTM): Encodes access history → Query vector
  2. Path Encoder (Tree-LSTM): Encodes each root-to-leaf path → Leaf embeddings (Keys/Values)
  3. Attention: Scores each leaf node based on query-key similarity
  
The model supports incremental state computation: when a new node is added
to the trie, only that node's LSTM state needs to be computed from its parent.
"""

import collections
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
    - Keys/Values come from Tree-LSTM encoded leaf states (not static cache line embeddings)
    - Supports incremental node state computation
    - Query is still from access history LSTM
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
        
        # --- History Encoder: LSTM over access history to produce Query ---
        self.history_lstm = nn.LSTMCell(history_embed_dim, hidden_size)
        
        # Project node embedding to history input dim (in case they differ)
        if node_embed_dim != history_embed_dim:
            self.history_proj = nn.Linear(node_embed_dim, history_embed_dim)
        else:
            self.history_proj = nn.Identity()
        
        # --- Path Encoder: Tree-LSTM for encoding root-to-leaf paths ---
        self.path_lstm = PathLSTMCell(node_embed_dim, hidden_size)
        
        # --- Attention: Query (history) attends over Keys (leaf states) ---
        # Using scaled dot-product attention for simplicity
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        # --- Scorer: maps attention context to eviction score ---
        self.scorer = nn.Linear(hidden_size, 1)
        
        # --- Reuse distance estimator ---
        self.reuse_distance_estimator = nn.Linear(hidden_size, 1)
    
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
        node_embed = self.node_embedder.embed_single(node_id, device)  # (1, embed_dim)
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
            node_id: ID of the currently accessed node (last node in accessed sequence)
            prev_state: Previous (h, c) from history LSTM. None for first step.
            
        Returns:
            Updated (h, c) for history LSTM
        """
        device = next(self.parameters()).device
        node_embed = self.node_embedder.embed_single(node_id, device)  # (1, embed_dim)
        history_input = self.history_proj(node_embed)  # (1, history_embed_dim)
        
        if prev_state is None:
            h = torch.zeros(1, self.hidden_size, device=device)
            c = torch.zeros(1, self.hidden_size, device=device)
            prev_state = (h, c)
        
        h_new, c_new = self.history_lstm(history_input, prev_state)
        return h_new, c_new
    
    def forward(
        self,
        history_state: torch.Tensor,
        leaf_states: List[torch.Tensor] = None,
        leaf_paths: List[Tuple[int, ...]] = None,
        inference: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eviction scores for leaf nodes.
        
        Mirrors the original Parrot model's design: a single forward() that
        handles both inference and training, controlled by an `inference` flag.
        loss() calls self() with inference=False.
        
        When inference=True:
            Uses pre-computed leaf_states from trie cache (incremental, fast).
        When inference=False:
            Re-encodes leaf_paths from scratch with gradient flowing through
            the path encoder and embedder (for training).
        
        Args:
            history_state: History LSTM hidden state, shape (1, hidden_size)
            leaf_states: List of N pre-computed leaf hidden states (inference mode),
                        each shape (1, hidden_size). Required when inference=True.
            leaf_paths: List of N root-to-leaf node ID tuples (training mode).
                        Required when inference=False.
            inference: If True, use cached leaf_states; if False, re-encode
                      leaf_paths with gradient.
        
        Returns:
            eviction_logits: shape (1, N) — higher means more likely to evict
            pred_reuse_distances: shape (1, N)
        """
        device = next(self.parameters()).device
        
        if inference:
            # Fast path: use pre-computed states from trie (no re-encoding)
            assert leaf_states is not None, "leaf_states required when inference=True"
            if len(leaf_states) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            keys = torch.cat(leaf_states, dim=0)  # (N, hidden_size)
        else:
            # Training path: re-encode each root-to-leaf path with gradient
            assert leaf_paths is not None, "leaf_paths required when inference=False"
            if len(leaf_paths) == 0:
                return torch.zeros(1, 0, device=device), torch.zeros(1, 0, device=device)
            encoded = []
            for path in leaf_paths:
                state = None
                for nid in path:
                    node_embed = self.node_embedder.embed_single(nid, device)
                    state = self.path_lstm(node_embed, state)
                if state is not None:
                    encoded.append(state[0])  # h component, (1, H)
                else:
                    encoded.append(torch.zeros(1, self.hidden_size, device=device))
            keys = torch.cat(encoded, dim=0)  # (N, hidden_size)
        
        query = self.query_proj(history_state)         # (1, hidden_size)
        proj_keys = self.key_proj(keys)               # (N, hidden_size)
        
        scale = self.hidden_size ** 0.5
        attn_logits = torch.matmul(query, proj_keys.T) / scale          # (1, N)
        scorer_logits = self.scorer(proj_keys).squeeze(-1).unsqueeze(0)  # (1, N)
        eviction_logits = scorer_logits - attn_logits                   # (1, N)
        
        pred_reuse_dist = self.reuse_distance_estimator(proj_keys).squeeze(-1).unsqueeze(0)
        
        return eviction_logits, pred_reuse_dist
    
    def initial_history_state(self, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized history state."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        return h, c
    
    def loss(self, snapshots) -> Dict[str, torch.Tensor]:
        """
        Compute training loss from snapshots collected by TrieTrainingCache.
        
        Mirrors original Parrot: loss() calls self() (forward) directly.
        forward(inference=False) re-encodes leaf paths with gradient.
        
        Args:
            snapshots: List of SimpleNamespace from TrieTrainingCache.get_snapshots().
                Each has .eviction_steps, where each step has:
                - leaf_paths: list of tuples (root-to-leaf node ID sequences)
                - oracle_target: int index of leaf to evict
                - history_state: (h, c) tensors or None
                - num_candidates: int
        
        Returns:
            Dict mapping loss name to scalar tensor.
        """
        device = next(self.parameters()).device
        eviction_losses = []
        
        for snapshot in snapshots:
            for step in snapshot.eviction_steps:
                if step.num_candidates < 2:
                    continue
                
                # History query (detached — history encoder not trained in this loss)
                if step.history_state is not None:
                    history_h = step.history_state[0].detach().to(device)
                else:
                    history_h = torch.zeros(1, self.hidden_size, device=device)
                
                # Forward with inference=False: re-encodes paths with gradient
                logits, _ = self(history_h, leaf_paths=step.leaf_paths, inference=False)
                
                target = torch.tensor([step.oracle_target], device=device)
                eviction_losses.append(F.cross_entropy(logits, target))
        
        losses = {}
        if eviction_losses:
            losses['eviction'] = torch.stack(eviction_losses).mean()
        else:
            losses['eviction'] = torch.tensor(0.0, device=device, requires_grad=True)
        return losses

    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: Optional[str] = None) -> 'TrieParrotModel':
        """
        Create model from config file.
        
        Args:
            config_path: Path to JSON config file
            checkpoint_path: Optional path to model checkpoint
            
        Returns:
            Initialized TrieParrotModel
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = cls(
            vocab_size=config['vocab_size'],
            node_embed_dim=config.get('node_embed_dim', 64),
            history_embed_dim=config.get('history_embed_dim', 64),
            hidden_size=config.get('hidden_size', 128),
            max_attention_history=config.get('max_attention_history', 30),
        )
        
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model
