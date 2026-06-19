#!/usr/bin/env python3
"""Verify lru-trie replay trains path-level history slots."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trie_model.model import TrieParrotModel


def make_step():
    step = SimpleNamespace()
    step.leaf_paths = [(1, 2, 3), (1, 4, 5)]
    step.oracle_distances = [1.0, float("inf")]
    step.oracle_target = 1
    step.num_candidates = 2
    step.microstep_history_paths = ((9,), (9, 8), (9, 8, 7))
    step.request_history_paths = ((1, 2, 3),)
    step.lru_features = (
        (1.0, 1.0, 1.0, 1.0, 3.0),
        (2.0, 2.0, 2.0, 2.0, 3.0),
    )
    return step


def test_path_history_replay_uses_path_lstm():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.train()

    snapshot = SimpleNamespace()
    snapshot.eviction_steps = [make_step()]

    losses = model.loss([snapshot])
    sum(losses.values()).backward()

    path_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "path_lstm" in name:
            path_grad += float(param.grad.abs().sum().item())

    assert path_grad > 0.0, "path_lstm should train through path-level history replay"
    assert all(
        "history_lstm" not in name and "history_proj" not in name
        for name, _ in model.named_parameters()
    )


if __name__ == "__main__":
    test_path_history_replay_uses_path_lstm()
    print("PATH HISTORY REPLAY TRAINING TEST PASSED")
