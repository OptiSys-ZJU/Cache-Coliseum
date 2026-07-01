#!/usr/bin/env python3
"""Verify PARROT-like microstep path-history timing for runtime and training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieNode
from cache.trie.trie_cache import TrieTrainingCache
from model.trie_model.model import TrieParrotModel


class RecordingModel(TrieParrotModel):
    def __init__(self):
        super().__init__(
            vocab_size=256,
            node_embed_dim=16,
            hidden_size=32,
            max_attention_history=16,
        )
        self.runtime_history_lengths = []

    def forward(
        self,
        microstep_history_memory,
        request_history_memory,
        lru_features,
        candidate_states=None,
        candidate_paths=None,
        lcp_features=None,
        inference=True,
    ):
        del lcp_features
        if isinstance(microstep_history_memory, list):
            self.runtime_history_lengths.append(len(microstep_history_memory))
        elif microstep_history_memory is None:
            self.runtime_history_lengths.append(0)
        else:
            self.runtime_history_lengths.append(int(microstep_history_memory.shape[0]))

        num_candidates = (
            len(candidate_states) if candidate_states is not None else len(candidate_paths)
        )
        assert request_history_memory is None or len(request_history_memory) >= 1
        assert len(lru_features) == num_candidates
        logits = torch.zeros(1, num_candidates, dtype=torch.float32)
        reuse = torch.zeros(1, num_candidates, dtype=torch.float32)
        return logits, reuse


class SnapshotRecordingModel(TrieParrotModel):
    def __init__(self):
        super().__init__(
            vocab_size=256,
            node_embed_dim=16,
            hidden_size=32,
            max_attention_history=16,
        )
        self.snapshot_microstep_history_paths = []

    def loss(
        self,
        snapshots,
        max_candidates=None,
        max_steps_per_snapshot=None,
    ):
        self.snapshot_microstep_history_paths = [
            tuple(step.microstep_history_paths)
            for snapshot in snapshots
            for step in snapshot.eviction_steps
        ]
        return super().loss(
            snapshots,
            max_candidates=max_candidates,
            max_steps_per_snapshot=max_steps_per_snapshot,
        )


def test_evictions_during_request_see_score_time_microstep_history():
    model = RecordingModel()
    model.eval()
    alg = TrieModelPredictAlgorithm(max_node_num=3, model=model)

    alg.access([1, 2])
    assert list(alg.microstep_history_path_window) == [(1,), (1, 2)]

    alg.access([3, 4, 5])
    assert model.runtime_history_lengths == [4, 5], (
        "evictions during request [3,4,5] should see the access-prefix "
        "history plus the current prefix available for scoring"
    )
    assert list(alg.microstep_history_path_window) == [
        (1,),
        (1, 2),
        (3,),
        (3, 4),
        (3, 4, 5),
    ]


def test_collect_microstep_history_paths_include_current_prefix_for_scoring():
    model = TrieParrotModel(
        vocab_size=256,
        node_embed_dim=16,
        hidden_size=32,
        max_attention_history=16,
    )
    model.eval()
    cache = TrieTrainingCache(max_node_num=4, model=model)
    sequences = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    for seq in sequences[:2]:
        cache.collect(seq)
    snapshot, _ = cache.collect(sequences[2])

    assert snapshot is not None, "expected microstep cache-state snapshots"
    step_prefixes = [tuple(step.current_path) for step in snapshot.eviction_steps]
    step_histories = [tuple(step.microstep_history_paths) for step in snapshot.eviction_steps]

    assert step_prefixes == [(5,), (5, 6)]
    assert [step.step_kind for step in snapshot.eviction_steps] == [
        "microstep_access",
        "microstep_access",
    ]
    assert step_histories == [
        ((1,), (1, 2), (3,), (3, 4), (5,)),
        ((1,), (1, 2), (3,), (3, 4), (5,), (5, 6)),
    ], "microstep scoring should see old history plus the current prefix"
    assert (5,) in step_histories[0]
    assert (5, 6) in step_histories[1]
    assert cache.alg.eviction_count == 2
    assert list(cache.alg.microstep_history_path_window) == [
        (1,),
        (1, 2),
        (3,),
        (3, 4),
        (5,),
        (5, 6),
    ]


def test_collect_to_loss_replays_same_leaf_history_snapshots():
    model = SnapshotRecordingModel()
    model.train()
    cache = TrieTrainingCache(max_node_num=4, model=model)
    sequences = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    for seq in sequences:
        cache.collect(seq)

    snapshots = cache.get_snapshots()
    assert snapshots, "expected at least one snapshot"

    expected_histories = [
        tuple(step.microstep_history_paths)
        for snapshot in snapshots
        for step in snapshot.eviction_steps
    ]

    losses = model.loss(snapshots)
    sum(losses.values()).backward()

    assert model.snapshot_microstep_history_paths == expected_histories, (
        "loss replay should consume the same prefix-history snapshots captured at collect() time"
    )

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


def test_protection_uses_current_prefix_not_future_suffix():
    alg = TrieModelPredictAlgorithm(max_node_num=20, model=None)

    for seq in ([1, 2, 9], [1, 2, 8], [1, 7]):
        this_node, insert_list = alg.__match__(seq)
        alg.__insert__(this_node, insert_list)

    protected_prefix = {
        TrieNode.get_path_tuple_from_node(node)
        for node in alg._get_protected_leaves([1, 2])
    }
    protected_full = {
        TrieNode.get_path_tuple_from_node(node)
        for node in alg._get_protected_leaves([1, 2, 9])
    }

    assert protected_prefix == set(), "current prefix should not protect future suffix leaves"
    assert protected_full == {(1, 2, 9)}, "full path protection should only appear once the suffix is visible"


def test_train_infer_history_lengths_align_for_same_sequence():
    runtime_model = RecordingModel()
    runtime_model.eval()
    alg = TrieModelPredictAlgorithm(max_node_num=4, model=runtime_model)

    train_model = TrieParrotModel(
        vocab_size=256,
        node_embed_dim=16,
        hidden_size=32,
        max_attention_history=16,
    )
    train_model.eval()
    cache = TrieTrainingCache(max_node_num=4, model=train_model)

    warmup = [[1, 2], [3, 4]]
    target = [5, 6]

    for seq in warmup:
        alg.access(seq)

    cache.load_future_accesses(warmup + [target])
    cache.set_model_prob(0.0)
    for seq in warmup:
        cache.collect(seq)

    snapshot, _ = cache.collect(target)
    assert snapshot is not None, "expected training microstep snapshots for target sequence"

    alg.access(target)

    runtime_lengths = list(runtime_model.runtime_history_lengths)
    snapshot_lengths = [len(step.microstep_history_paths) for step in snapshot.eviction_steps]

    assert snapshot_lengths == [5, 6]
    assert runtime_lengths == [5, 6], (
        "runtime evictions and microstep training snapshots should share the "
        "same score-time history visibility"
    )


def test_history_window_bounding_matches_runtime_and_replay():
    model = SnapshotRecordingModel()
    model.train()
    cache = TrieTrainingCache(max_node_num=4, model=model)
    sequences = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    for seq in sequences:
        cache.collect(seq)

    snapshots = cache.get_snapshots()
    assert snapshots, "expected snapshots after overflowing bounded history"

    losses = model.loss(snapshots)
    sum(losses.values()).backward()

    for microstep_history_paths in model.snapshot_microstep_history_paths:
        assert len(microstep_history_paths) <= model.max_attention_history, (
            "replayed history paths should respect the same bounded window as runtime memory"
        )


if __name__ == "__main__":
    test_evictions_during_request_see_score_time_microstep_history()
    test_collect_microstep_history_paths_include_current_prefix_for_scoring()
    test_collect_to_loss_replays_same_leaf_history_snapshots()
    test_protection_uses_current_prefix_not_future_suffix()
    test_train_infer_history_lengths_align_for_same_sequence()
    test_history_window_bounding_matches_runtime_and_replay()
    print("STEP-WISE HISTORY UPDATE TESTS PASSED")
