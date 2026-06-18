#!/usr/bin/env python3
"""Verify incremental Tree-LSTM state uses parent hidden_state correctly."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trie_model.model import TrieParrotModel
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm

model = TrieParrotModel(vocab_size=100, node_embed_dim=16, hidden_size=32)
model.eval()

alg = TrieModelPredictAlgorithm(max_node_num=20, model=model)

# Insert path [1, 2, 3]
alg.access([1, 2, 3])

# Walk tree: root -> 1 -> 2 -> 3
node1 = alg.root_node.children[1]
node2 = node1.children[2]
node3 = node2.children[3]

# Verify all nodes have hidden_state
assert node1.hidden_state is not None, 'Node 1 should have hidden_state'
assert node2.hidden_state is not None, 'Node 2 should have hidden_state'
assert node3.hidden_state is not None, 'Node 3 should have hidden_state'

# Verify incremental: node2's state should use node1's state, not zero
with torch.no_grad():
    h_from_scratch, _ = model.compute_node_state(2, None)  # no parent
    h_from_parent, _ = model.compute_node_state(2, node1.hidden_state)  # with parent

cached_h = node2.hidden_state[0]
assert torch.allclose(cached_h, h_from_parent, atol=1e-6), \
    'Node2 state should match compute_node_state(2, node1.hidden_state)'
assert not torch.allclose(cached_h, h_from_scratch, atol=1e-6), \
    'Node2 state should NOT match compute_node_state(2, None) — parent state must be used'

# Also verify node3 uses node2's state
with torch.no_grad():
    h3_expected, _ = model.compute_node_state(3, node2.hidden_state)
assert torch.allclose(node3.hidden_state[0], h3_expected, atol=1e-6), \
    'Node3 state should use node2.hidden_state'

print('INCREMENTAL STATE TEST PASSED: all nodes use parent hidden_state correctly')
