#!/usr/bin/env python3
"""Verify history replay gives history_lstm non-zero gradients."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from model.trie_model.model import TrieParrotModel


def make_step():
    step = SimpleNamespace()
    step.leaf_paths = [(1, 2, 3), (1, 4, 5)]
    step.oracle_target = 0
    step.num_candidates = 2
    step.history_tokens = (9, 8, 7, 6)
    return step


def test_history_lstm_gets_gradients_from_replay():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.train()

    snapshot = SimpleNamespace()
    snapshot.eviction_steps = [make_step()]

    losses = model.loss([snapshot])
    losses["eviction"].backward()

    history_grad = 0.0
    for name, param in model.named_parameters():
        if "history_lstm" in name and param.grad is not None:
            history_grad += float(param.grad.abs().sum().item())

    assert history_grad > 0.0, "history_lstm should receive gradients via replayed history tokens"


if __name__ == "__main__":
    test_history_lstm_gets_gradients_from_replay()
    print("HISTORY LSTM REPLAY TRAINING TEST PASSED")
