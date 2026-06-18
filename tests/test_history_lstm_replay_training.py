#!/usr/bin/env python3
"""Verify v1 replay trains path-level history, not the legacy history_lstm."""
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
    step.history_paths = ((9,), (9, 8), (9, 8, 7))
    return step


def test_path_history_replay_uses_path_lstm_not_history_lstm():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.train()

    snapshot = SimpleNamespace()
    snapshot.eviction_steps = [make_step()]

    losses = model.loss([snapshot])
    sum(losses.values()).backward()

    path_grad = 0.0
    history_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "path_lstm" in name:
            path_grad += float(param.grad.abs().sum().item())
        if "history_lstm" in name:
            history_grad += float(param.grad.abs().sum().item())

    assert path_grad > 0.0, "path_lstm should train through path-level history replay"
    assert history_grad == 0.0, "Trie-PARROT v1 should not use legacy history_lstm"


if __name__ == "__main__":
    test_path_history_replay_uses_path_lstm_not_history_lstm()
    print("PATH HISTORY REPLAY TRAINING TEST PASSED")
