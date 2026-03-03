"""
Tree-LSTM / Path-LSTM module for encoding tree paths.

This module implements the core LSTM cell that processes tree paths in a
top-down manner. For each node, the hidden state is computed based on:
    h_child = LSTM(h_parent, embedding(node_id))

This allows incremental computation - when a new node is added to a path,
only that node's state needs to be computed.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PathLSTMCell(nn.Module):
    """
    LSTM cell for path-based tree encoding.
    
    Given a parent's hidden state and the current node's embedding,
    computes the current node's hidden state.
    
    This is essentially a standard LSTMCell, but named specifically for
    the tree-path use case and optimized for single-step computation.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize the PathLSTMCell.
        
        Args:
            input_size: Dimension of node embeddings
            hidden_size: Dimension of hidden state (h and c)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Standard LSTM gates: input, forget, cell, output
        # Combined into single linear for efficiency
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.gates.weight)
        # Initialize forget gate bias to 1 for better gradient flow
        nn.init.zeros_(self.gates.bias)
        # Set forget gate bias to 1
        self.gates.bias.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(
        self, 
        node_embedding: torch.Tensor, 
        parent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden state for current node.
        
        Args:
            node_embedding: Embedding of current node, shape (batch, input_size)
            parent_state: Tuple of (h, c) from parent node, each shape (batch, hidden_size)
                         If None, uses zero initialization (for root node)
        
        Returns:
            Tuple of (h, c) for current node, each shape (batch, hidden_size)
        """
        batch_size = node_embedding.size(0)
        device = node_embedding.device
        
        # Initialize parent state if not provided (root node case)
        if parent_state is None:
            h_parent = torch.zeros(batch_size, self.hidden_size, device=device)
            c_parent = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_parent, c_parent = parent_state
        
        # Concatenate input and hidden state
        combined = torch.cat([node_embedding, h_parent], dim=1)
        
        # Compute all gates at once
        gates = self.gates(combined)
        
        # Split into individual gates
        i_gate = torch.sigmoid(gates[:, :self.hidden_size])
        f_gate = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g_gate = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o_gate = torch.sigmoid(gates[:, 3*self.hidden_size:])
        
        # Compute new cell state and hidden state
        c_new = f_gate * c_parent + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create zero-initialized hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h, c), each shape (batch_size, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c


class PathEncoder(nn.Module):
    """
    Encodes a sequence of node IDs into a hidden state.
    
    This processes a path from root to leaf, computing LSTM states
    incrementally for each node.
    
    Note: TrieParrotModel currently uses PathLSTMCell + NodeEmbedder directly
    for finer-grained control. This class is provided as a convenience utility
    for standalone path encoding (e.g., testing or alternative architectures).
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        """
        Initialize PathEncoder.
        
        Args:
            vocab_size: Size of node ID vocabulary
            embed_dim: Dimension of node embeddings
            hidden_size: Dimension of LSTM hidden state
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = PathLSTMCell(embed_dim, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(
        self, 
        path: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a path (sequence of node IDs).
        
        Args:
            path: Tensor of node IDs, shape (batch, seq_len)
            initial_state: Optional initial (h, c) state
            
        Returns:
            Final (h, c) state after processing entire path
        """
        batch_size, seq_len = path.size()
        
        state = initial_state
        for t in range(seq_len):
            node_ids = path[:, t]
            node_embed = self.embedding(node_ids)
            state = self.lstm_cell(node_embed, state)
        
        return state
    
    def forward_incremental(
        self,
        node_id: torch.Tensor,
        parent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state for a single node (incremental update).
        
        Args:
            node_id: Single node ID, shape (batch,) or scalar
            parent_state: Parent's (h, c) state
            
        Returns:
            (h, c) state for this node
        """
        if node_id.dim() == 0:
            node_id = node_id.unsqueeze(0)
        node_embed = self.embedding(node_id)
        return self.lstm_cell(node_embed, parent_state)
