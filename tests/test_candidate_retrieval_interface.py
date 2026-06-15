#!/usr/bin/env python3
"""Smoke tests for candidate-conditioned retrieval interfaces."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


def make_model():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.eval()
    return model


def test_forward_shapes():
    model = make_model()
    history_memory = torch.randn(2, model.hidden_size)
    candidate_paths = [(1, 2, 3), (1, 2, 4)]

    logits, reuse = model.forward(
        history_memory,
        candidate_paths=candidate_paths,
        inference=False,
    )

    assert logits.shape == (1, 2), logits.shape
    assert reuse.shape == (1, 2), reuse.shape

def test_training_snapshot_carries_request_metadata():
    model = make_model()
    cache = TrieTrainingCache(max_node_num=5, model=model)
    sequences = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 2, 3],
        [1, 2, 6],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    snapshot = None
    for seq in sequences:
        maybe_snapshot, _ = cache.collect(seq)
        if maybe_snapshot is not None:
            snapshot = maybe_snapshot
            break

    assert snapshot is not None, "expected at least one eviction snapshot"
    assert snapshot.eviction_steps, "expected eviction steps"
    assert isinstance(snapshot.sequence, tuple)
    assert snapshot.sequence == (1, 2, 6)


test_forward_shapes()
test_training_snapshot_carries_request_metadata()

print("CANDIDATE RETRIEVAL INTERFACE TESTS PASSED")
