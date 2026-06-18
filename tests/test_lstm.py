#!/usr/bin/env python3
"""Test Tree-LSTM module."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trie_model.tree_lstm import PathLSTMCell, PathEncoder

print('Testing PathLSTMCell...')
cell = PathLSTMCell(input_size=64, hidden_size=128)
batch_size = 4

node_embed = torch.randn(batch_size, 64)
h, c = cell(node_embed, None)
print(f'  Output shape (no parent): h={h.shape}, c={c.shape}')
assert h.shape == (batch_size, 128), 'Wrong h shape'
assert c.shape == (batch_size, 128), 'Wrong c shape'

h2, c2 = cell(node_embed, (h, c))
print(f'  Output shape (with parent): h={h2.shape}, c={c2.shape}')

print('\nTesting PathEncoder...')
encoder = PathEncoder(vocab_size=1000, embed_dim=64, hidden_size=128)
path = torch.randint(0, 1000, (batch_size, 5))
h_final, c_final = encoder(path)
print(f'  Full path encoding: h={h_final.shape}, c={c_final.shape}')

node_id = torch.tensor([42])
h_inc, c_inc = encoder.forward_incremental(node_id, None)
print(f'  Incremental encoding: h={h_inc.shape}')

print('\nTask 3.1 verification passed!')
