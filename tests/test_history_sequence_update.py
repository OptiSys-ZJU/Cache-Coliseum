#!/usr/bin/env python3
"""Verify history LSTM updates use the cache-visible request sequence."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from cache.trie.trie_algorithms import TrieModelPredictAlgorithm
from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


def assert_state_close(actual, expected, label):
    assert actual is not None, f"{label}: actual state should not be None"
    assert expected is not None, f"{label}: expected state should not be None"
    actual_h, actual_c = actual
    expected_h, expected_c = expected
    assert torch.allclose(actual_h, expected_h, atol=1e-6), (
        f"{label}: hidden state mismatch"
    )
    assert torch.allclose(actual_c, expected_c, atol=1e-6), (
        f"{label}: cell state mismatch"
    )


def encode_sequence(model, sequence, prev_state=None):
    state = prev_state
    with torch.no_grad():
        for node_id in sequence:
            state = model.encode_history_step(node_id, state)
    return state


model = TrieParrotModel(vocab_size=100, node_embed_dim=16, hidden_size=32)
model.eval()

sequence = [11, 22, 33]
long_sequence = [7, 8, 9]

# Inference path: TrieModelPredictAlgorithm.access()
alg = TrieModelPredictAlgorithm(max_node_num=20, model=model)
alg.access(sequence)
expected_alg_state = encode_sequence(model, sequence)
assert_state_close(alg.history_state, expected_alg_state, "predict algorithm")

# Training path: TrieTrainingCache.collect()
train_cache = TrieTrainingCache(max_node_num=20, model=model)
train_cache.collect(sequence)
expected_cache_state = encode_sequence(model, sequence)
assert_state_close(
    train_cache.alg.history_state,
    expected_cache_state,
    "training cache",
)

# Long requests should use the cache-visible prefix on the training path, which
# mirrors the current inference-side truncation policy.
small_train_cache = TrieTrainingCache(max_node_num=2, model=model)
small_train_cache.collect(long_sequence)
expected_small_cache_state = encode_sequence(model, long_sequence[:2])
assert_state_close(
    small_train_cache.alg.history_state,
    expected_small_cache_state,
    "training cache long request",
)

print(
    "HISTORY SEQUENCE UPDATE TEST PASSED: cache-visible request prefixes are "
    "encoded into history state"
)
