#!/usr/bin/env python3
"""Verify access prefixes become path-level history slots."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cache.trie.trie_algorithms import TrieModelPredictAlgorithm
from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


def assert_microstep_history_paths(actual, expected, label):
    assert list(actual) == expected, f"{label}: expected {expected}, got {list(actual)}"


model = TrieParrotModel(vocab_size=100, node_embed_dim=16, hidden_size=32)
model.eval()

first_sequence = [1, 2]
second_sequence = [3, 4]
long_sequence = [7, 8, 9]

# Inference path: TrieModelPredictAlgorithm.access()
alg = TrieModelPredictAlgorithm(max_node_num=20, model=model)
alg.access(first_sequence)
assert_microstep_history_paths(
    alg.microstep_history_path_window,
    [(1,), (1, 2)],
    "predict algorithm",
)
alg.access(second_sequence)
assert_microstep_history_paths(
    alg.microstep_history_path_window,
    [(1,), (1, 2), (3,), (3, 4)],
    "predict algorithm second request",
)
assert not hasattr(alg, "history_state")

# Training path: TrieTrainingCache.collect()
train_cache = TrieTrainingCache(max_node_num=20, model=model)
train_cache.collect(first_sequence)
assert_microstep_history_paths(
    train_cache.alg.microstep_history_path_window,
    [(1,), (1, 2)],
    "training cache",
)
train_cache.collect(second_sequence)
assert_microstep_history_paths(
    train_cache.alg.microstep_history_path_window,
    [(1,), (1, 2), (3,), (3, 4)],
    "training cache second request",
)
assert not hasattr(train_cache.alg, "history_state")

# Long requests should record each cache-visible prefix on both inference and
# training paths.
small_alg = TrieModelPredictAlgorithm(max_node_num=2, model=model)
small_alg.access(long_sequence)
assert_microstep_history_paths(
    small_alg.microstep_history_path_window,
    [(7,), (7, 8)],
    "predict algorithm long request",
)

small_train_cache = TrieTrainingCache(max_node_num=2, model=model)
small_train_cache.collect(long_sequence)
assert_microstep_history_paths(
    small_train_cache.alg.microstep_history_path_window,
    [(7,), (7, 8)],
    "training cache long request",
)

print(
    "HISTORY SEQUENCE UPDATE TEST PASSED: access prefixes are encoded "
    "as path-level history slots"
)
