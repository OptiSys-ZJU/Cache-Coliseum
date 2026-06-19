#!/usr/bin/env python3
"""Verify collect() -> loss() closes the path-history replay loop."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


def test_collect_to_loss_replay():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.train()

    cache = TrieTrainingCache(max_node_num=5, model=model)
    sequences = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 2, 3],
        [1, 2, 6],
        [1, 2, 4],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    for seq in sequences:
        cache.collect(seq)

    snapshots = cache.get_snapshots()
    assert snapshots, "expected at least one snapshot"
    assert any(
        getattr(step, "microstep_history_paths", None)
        for snapshot in snapshots
        for step in snapshot.eviction_steps
    )
    assert all(
        hasattr(step, "oracle_distances")
        for snapshot in snapshots
        for step in snapshot.eviction_steps
    )

    losses = model.loss(snapshots)
    sum(losses.values()).backward()

    path_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "path_lstm" in name:
            path_grad += float(param.grad.abs().sum().item())

    assert path_grad > 0.0, "path_lstm should receive gradients through collect()->loss() replay"
    assert all(
        "history_lstm" not in name and "history_proj" not in name
        for name, _ in model.named_parameters()
    )


if __name__ == "__main__":
    test_collect_to_loss_replay()
    print("TRAINING CACHE REPLAY LOSS TEST PASSED")
