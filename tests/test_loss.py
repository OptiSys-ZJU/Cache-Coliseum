#!/usr/bin/env python3
"""Test TrieParrotModel.loss() with Trie-PARROT v1 gradient flow."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from model.trie_model.model import TrieParrotModel


model = TrieParrotModel(vocab_size=100, node_embed_dim=32, hidden_size=64)
model.train()

# Create mock training snapshots as TrieTrainingCache would produce.
step1 = SimpleNamespace()
step1.leaf_paths = [(1, 2, 3), (1, 2, 4), (1, 2, 5)]
step1.oracle_target = 2
step1.num_candidates = 3
step1.history_paths = ((9,), (9, 9), (9, 9, 9))
step1.oracle_distances = [1.0, 2.0, float("inf")]

step2 = SimpleNamespace()
step2.leaf_paths = [(1, 2, 4), (1, 2, 5)]
step2.oracle_target = 1
step2.num_candidates = 2
step2.history_paths = ((8,), (8, 8))
step2.oracle_distances = [3.0, float("inf")]

snapshot = SimpleNamespace()
snapshot.eviction_steps = [step1, step2]

# Compute loss
losses = model.loss([snapshot])
print(f"Loss: {losses}")
assert "reuse" in losses
assert "ranking" in losses
assert "ce" in losses
assert losses["ranking"].requires_grad
assert losses["reuse"].requires_grad
assert torch.isfinite(losses["reuse"]), "reuse loss should stay finite with inf oracle distances"
assert losses["ce"].item() == 0.0, "CE should be disabled by default"
print(f"  ranking loss = {losses['ranking'].item():.4f}")
print(f"  reuse loss = {losses['reuse'].item():.4f}")

# Backward pass
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
total_loss = sum(losses.values())
total_loss.backward()

# Check gradients exist
grad_count = 0
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        grad_count += 1
print(f"  Parameters with non-zero gradients: {grad_count}")
assert grad_count > 0, "Expected some gradients"

# Verify path_lstm gets gradients from both candidate and path-history replay.
path_lstm_has_grad = False
for name, param in model.named_parameters():
    if "path_lstm" in name and param.grad is not None and param.grad.abs().sum() > 0:
        path_lstm_has_grad = True
        break
assert path_lstm_has_grad, "path_lstm should receive gradients through loss"
print("  path_lstm has gradients: True")

# Verify legacy history_lstm is not used in Trie-PARROT v1.
history_lstm_has_grad = False
for name, param in model.named_parameters():
    if "history_lstm" in name and param.grad is not None and param.grad.abs().sum() > 0:
        history_lstm_has_grad = True
        break
assert not history_lstm_has_grad, "history_lstm should not receive gradients in Trie-PARROT v1"
print("  history_lstm has gradients: False")

# Verify scorer gets gradients.
scorer_has_grad = False
for name, param in model.named_parameters():
    if "scorer" in name and param.grad is not None and param.grad.abs().sum() > 0:
        scorer_has_grad = True
        break
assert scorer_has_grad, "scorer should receive gradients"
print("  scorer has gradients: True")

# Test with empty snapshots.
empty_losses = model.loss([])
assert empty_losses["ranking"].item() == 0.0
assert empty_losses["reuse"].item() == 0.0
assert empty_losses["ce"].item() == 0.0
print("  empty snapshot loss = 0.0: OK")

print("\nTrieParrotModel.loss verification passed")
