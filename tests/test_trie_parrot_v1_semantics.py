#!/usr/bin/env python3
"""Regression checks for the first Trie-PARROT path-history design."""
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from cache.trie.oracle import PrefixFutureOracle
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieNode
from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


def _insert_raw(alg, sequence):
    this_node, insert_list = alg.__match__(sequence)
    alg.__insert__(this_node, insert_list)


def _node_for_path(alg, sequence):
    node = alg.root_node
    for token in sequence:
        node = node.children[token]
    return node


def test_prefix_oracle_remains_request_clock():
    sequences = [
        [1, 2, 3],
        [1, 9],
        [1, 2, 3],
    ]
    oracle = PrefixFutureOracle(sequences)
    oracle.consume_current(sequences[0], 0)

    assert oracle.reuse_distance((1,), 0) == 1
    assert oracle.reuse_distance((1, 2), 0) == 2
    assert oracle.reuse_distance((1, 2, 3), 0) == 2


def test_snapshot_carries_oracle_distances_and_argmax_target():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    cache = TrieTrainingCache(max_node_num=4, model=model)
    sequences = [
        [1, 2],
        [3, 4],
        [1, 2],
        [5],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    snapshot = None
    for seq in sequences:
        maybe_snapshot, _ = cache.collect(seq)
        if maybe_snapshot is not None:
            snapshot = maybe_snapshot
            break

    assert snapshot is not None
    step = snapshot.eviction_steps[0]
    assert hasattr(step, "oracle_distances")
    assert len(step.oracle_distances) == step.num_candidates == len(step.leaf_paths)
    assert step.oracle_target == max(
        range(len(step.oracle_distances)),
        key=lambda idx: step.oracle_distances[idx],
    )


def test_collect_buffers_request_state_snapshot_once_without_prefix_steps():
    cache = TrieTrainingCache(max_node_num=4, model=None)
    sequences = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    returned_snapshot = None
    for seq in sequences:
        maybe_snapshot, _ = cache.collect(seq)
        if maybe_snapshot is not None:
            returned_snapshot = maybe_snapshot

    assert returned_snapshot is not None
    returned_paths = [
        step.current_path for step in returned_snapshot.eviction_steps
    ]
    assert returned_paths == [(5, 6)]
    assert returned_snapshot.eviction_steps[0].step_kind == "request_state"
    assert cache.alg.eviction_count == 2

    buffered = cache.get_snapshots()
    assert len(buffered) == 2
    buffered_paths = [
        step.current_path for snapshot in buffered for step in snapshot.eviction_steps
    ]
    assert buffered_paths == [(3, 4), (5, 6)]


def test_request_snapshot_uses_pre_access_live_leaf_set():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.eval()
    cache = TrieTrainingCache(max_node_num=10, model=model)
    sequences = [
        [1],
        [1, 2],
    ]
    cache.load_future_accesses(sequences)

    first_snapshot, _ = cache.collect(sequences[0])
    assert first_snapshot is None, "empty cache has no request-level candidates"

    second_snapshot, _ = cache.collect(sequences[1])
    assert second_snapshot is not None
    step = second_snapshot.eviction_steps[0]

    assert step.step_kind == "request_state"
    assert [tuple(path) for path in step.leaf_paths] == [(1,)]
    assert tuple(step.current_path) == (1, 2)
    assert list(cache.alg.history_path_window) == [(1,), (1, 2)]


def test_hit_without_eviction_still_collects_request_snapshot():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.eval()
    cache = TrieTrainingCache(max_node_num=10, model=model)
    sequences = [
        [1, 2],
        [1, 2],
    ]
    cache.load_future_accesses(sequences)

    cache.collect(sequences[0])
    snapshot, hit = cache.collect(sequences[1])

    assert hit
    assert snapshot is not None
    assert cache.alg.eviction_count == 0
    step = snapshot.eviction_steps[0]
    assert step.step_kind == "request_state"
    assert tuple(step.current_path) == (1, 2)
    assert [tuple(path) for path in step.leaf_paths] == [(1, 2)]
    assert tuple(step.history_paths) == ((1, 2),)


def test_pre_access_snapshot_labels_current_hit_as_immediate_reuse():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.eval()
    cache = TrieTrainingCache(max_node_num=10, model=model)
    sequences = [
        [1],
        [2],
        [1],
        [2],
    ]
    cache.load_future_accesses(sequences)

    cache.collect(sequences[0])
    cache.collect(sequences[1])
    snapshot, hit = cache.collect(sequences[2])

    assert hit
    assert snapshot is not None
    step = snapshot.eviction_steps[0]
    distances_by_path = {
        tuple(path): distance
        for path, distance in zip(step.leaf_paths, step.oracle_distances)
    }

    assert distances_by_path[(1,)] == 0
    assert distances_by_path[(2,)] == 1
    assert tuple(step.leaf_paths[step.oracle_target]) == (2,)


def test_continuous_eviction_adds_parent_candidate_and_recomputes_distances():
    cache = TrieTrainingCache(max_node_num=10, model=None)
    for seq in ([1, 2], [4], [5]):
        _insert_raw(cache.alg, seq)

    snapshot = cache._evict_and_collect(
        evict_num=2,
        this_node=cache.alg.root_node,
        current_path=[9],
        step_index=0,
    )

    assert len(snapshot.eviction_steps) == 2
    assert all(step.step_kind == "eviction_decision" for step in snapshot.eviction_steps)
    second_step_paths = [tuple(path) for path in snapshot.eviction_steps[1].leaf_paths]
    assert (1,) in second_step_paths, "deleting [1,2] should make parent [1] a candidate"
    assert len(snapshot.eviction_steps[1].oracle_distances) == len(second_step_paths)


def test_eviction_does_not_append_history():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    cache = TrieTrainingCache(max_node_num=10, model=model)
    for seq in ([1, 2], [3], [4]):
        _insert_raw(cache.alg, seq)

    cache.alg._record_history_leaf(_node_for_path(cache.alg, [3]))
    before_paths = tuple(cache.alg.history_path_window)
    before_len = len(cache.alg.history_hidden_states)

    cache._evict_and_collect(
        evict_num=1,
        this_node=cache.alg.root_node,
        current_path=[8],
        step_index=0,
    )

    assert tuple(cache.alg.history_path_window) == before_paths
    assert len(cache.alg.history_hidden_states) == before_len


def test_request_touch_appends_leaf_history_slot():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    alg = TrieModelPredictAlgorithm(max_node_num=10, model=model)

    alg.access([1, 2, 3])

    assert list(alg.history_path_window) == [(1, 2, 3)]
    assert len(alg.history_hidden_states) == 1


def test_candidate_is_query_only_not_direct_scorer_input():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=8)
    model.eval()

    history_memory = torch.randn(1, model.hidden_size)
    candidate_states = [
        torch.randn(1, model.hidden_size),
        torch.randn(1, model.hidden_size),
    ]

    with torch.no_grad():
        logits, pred_reuse = model.forward(
            history_memory,
            candidate_states=candidate_states,
            inference=True,
        )

    assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)
    assert torch.allclose(pred_reuse[:, 0], pred_reuse[:, 1], atol=1e-6)


def test_candidate_history_concat_scorer_uses_candidate_directly():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        candidate_scorer_mode="candidate_history_concat",
    )
    model.eval()

    history_memory = torch.randn(1, model.hidden_size)
    candidate_states = [
        torch.randn(1, model.hidden_size),
        torch.randn(1, model.hidden_size),
    ]

    with torch.no_grad():
        logits, pred_reuse = model.forward(
            history_memory,
            candidate_states=candidate_states,
            inference=True,
        )

    assert not torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)
    assert not torch.allclose(pred_reuse[:, 0], pred_reuse[:, 1], atol=1e-6)


def test_loss_handles_finite_and_inf_oracle_distances_without_nan():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=32,
        reuse_distance_log_cap=5.0,
    )
    model.train()

    step = SimpleNamespace(
        leaf_paths=[(1, 2), (3,), (4, 5)],
        oracle_distances=[1.0, float("inf"), 3.0],
        oracle_target=1,
        history_paths=((9,), (9, 8)),
        num_candidates=3,
    )
    snapshot = SimpleNamespace(eviction_steps=[step])

    losses = model.loss([snapshot])
    total = sum(losses.values())

    assert torch.isfinite(losses["ranking"])
    assert torch.isfinite(losses["reuse"])
    assert torch.isfinite(total)

    total.backward()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )
    assert math.isfinite(float(total.item()))


if __name__ == "__main__":
    test_prefix_oracle_remains_request_clock()
    test_snapshot_carries_oracle_distances_and_argmax_target()
    test_collect_buffers_request_state_snapshot_once_without_prefix_steps()
    test_request_snapshot_uses_pre_access_live_leaf_set()
    test_hit_without_eviction_still_collects_request_snapshot()
    test_pre_access_snapshot_labels_current_hit_as_immediate_reuse()
    test_continuous_eviction_adds_parent_candidate_and_recomputes_distances()
    test_eviction_does_not_append_history()
    test_request_touch_appends_leaf_history_slot()
    test_candidate_is_query_only_not_direct_scorer_input()
    test_candidate_history_concat_scorer_uses_candidate_directly()
    test_loss_handles_finite_and_inf_oracle_distances_without_nan()
    print("TRIE-PARROT V1 SEMANTICS TESTS PASSED")
