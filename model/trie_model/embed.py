"""
Node embedding module for Tree-LSTM cache model.

Provides embedding layers for converting node IDs (token IDs / item IDs)
into dense vector representations used as input to the PathLSTMCell.
"""

import torch
import torch.nn as nn
from typing import List, Union


class NodeEmbedder(nn.Module):
    """
    Embedding layer for trie node IDs.
    
    Maps integer node IDs to dense embedding vectors. Supports both
    single node and batch operations.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = None):
        """
        Initialize NodeEmbedder.
        
        Args:
            vocab_size: Number of unique node IDs in vocabulary
            embed_dim: Dimension of embedding vectors
            padding_idx: Optional padding index (embeddings at this index 
                        will be zero and not updated during training)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.embedding.weight.data)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
    
    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed node IDs.
        
        Args:
            node_ids: Tensor of integer node IDs, any shape
            
        Returns:
            Embeddings tensor with shape (*node_ids.shape, embed_dim)
        """
        return self.embedding(node_ids)
    
    def embed_single(self, node_id: int, device: torch.device = None) -> torch.Tensor:
        """
        Embed a single node ID (convenience method for incremental computation).
        
        Args:
            node_id: Single integer node ID
            device: Device for the tensor
            
        Returns:
            Embedding tensor of shape (1, embed_dim)
        """
        if device is None:
            device = self.embedding.weight.device
        ids = torch.tensor([node_id], device=device)
        return self.embedding(ids)
