#!/usr/bin/env python3
"""Test TrieParrotModel.loss() with gradient flow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from types import SimpleNamespace
from model.trie_model.model import TrieParrotModel

model = TrieParrotModel(vocab_size=100, node_embed_dim=32, hidden_size=64)
model.train()

# Create mock training snapshots (as TrieTrainingCache would produce)
step1 = SimpleNamespace()
step1.leaf_paths = [(1, 2, 3), (1, 2, 4), (1, 2, 5)]  # 3 candidates
step1.oracle_target = 2  # evict leaf at index 2 (path 1→2→5)
step1.num_candidates = 3
step1.history_tokens = (9, 9, 9)

step2 = SimpleNamespace()
step2.leaf_paths = [(1, 2, 4), (1, 2, 5)]  # 2 candidates after step1
step2.oracle_target = 1
step2.num_candidates = 2
step2.history_tokens = (8, 8)

snapshot = SimpleNamespace()
snapshot.eviction_steps = [step1, step2]

# Compute loss
losses = model.loss([snapshot])
print(f'Loss: {losses}')
assert 'eviction' in losses
assert losses['eviction'].requires_grad
print(f'  eviction loss = {losses["eviction"].item():.4f}')

# Backward pass
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
total_loss = losses['eviction']
total_loss.backward()

# Check gradients exist
grad_count = 0
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        grad_count += 1
print(f'  Parameters with non-zero gradients: {grad_count}')
assert grad_count > 0, 'Expected some gradients'

# Verify path_encoder gets gradients (key requirement)
path_lstm_has_grad = False
for name, param in model.named_parameters():
    if 'path_lstm' in name and param.grad is not None and param.grad.abs().sum() > 0:
        path_lstm_has_grad = True
        break
assert path_lstm_has_grad, 'path_lstm should receive gradients through loss'
print('  path_lstm has gradients: True')

# Verify scorer gets gradients
scorer_has_grad = False
for name, param in model.named_parameters():
    if 'scorer' in name and param.grad is not None and param.grad.abs().sum() > 0:
        scorer_has_grad = True
        break
assert scorer_has_grad, 'scorer should receive gradients'
print('  scorer has gradients: True')

# Test with empty snapshots
empty_losses = model.loss([])
assert empty_losses['eviction'].item() == 0.0
print('  empty snapshot loss = 0.0: OK')

print('\nTask 3.7 (TrieParrotModel.loss) verification passed!')
