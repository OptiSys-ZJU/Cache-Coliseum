#!/usr/bin/env python3
"""Verify PARROT-like step-wise history timing for trie runtime and training."""
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
        history_memory,
        candidate_states=None,
        candidate_paths=None,
        inference=True,
    ):
        if isinstance(history_memory, list):
            self.runtime_history_lengths.append(len(history_memory))
        elif history_memory is None:
            self.runtime_history_lengths.append(0)
        else:
            self.runtime_history_lengths.append(int(history_memory.shape[0]))

        num_candidates = (
            len(candidate_states) if candidate_states is not None else len(candidate_paths)
        )
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
        self.snapshot_history_tokens = []

    def loss(
        self,
        snapshots,
        max_candidates=None,
        max_steps_per_snapshot=None,
    ):
        self.snapshot_history_tokens = [
            tuple(step.history_tokens)
            for snapshot in snapshots
            for step in snapshot.eviction_steps
        ]
        return super().loss(
            snapshots,
            max_candidates=max_candidates,
            max_steps_per_snapshot=max_steps_per_snapshot,
        )


def test_step_i_sees_only_prior_prefix_and_step_i_plus_1_sees_h_i():
    model = RecordingModel()
    model.eval()
    alg = TrieModelPredictAlgorithm(max_node_num=3, model=model)

    alg.access([1, 2])
    assert list(alg.history_token_window) == [1, 2]

    alg.access([3, 4, 5])
    assert model.runtime_history_lengths == [3, 4], (
        "with capacity 3, eviction before block 4 should already see block 3, "
        "and eviction before block 5 should then see block 4 as well"
    )
    assert list(alg.history_token_window) == [1, 2, 3, 4, 5][-model.max_attention_history :]


def test_collect_snapshot_history_tokens_match_micro_step_prefixes():
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

    snapshot = None
    for seq in sequences:
        maybe_snapshot, _ = cache.collect(seq)
        if maybe_snapshot is not None:
            snapshot = maybe_snapshot
            break

    assert snapshot is not None, "expected an eviction snapshot"
    step_prefixes = [tuple(step.current_path) for step in snapshot.eviction_steps]
    step_histories = [tuple(step.history_tokens) for step in snapshot.eviction_steps]

    assert step_prefixes == [(5,), (5, 6)]
    assert step_histories == [(1, 2, 3, 4), (1, 2, 3, 4, 5)], (
        "step i history should stop before its own token, while step i+1 can see step i"
    )


def test_collect_to_loss_replays_same_micro_step_history_prefixes():
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
        tuple(step.history_tokens)
        for snapshot in snapshots
        for step in snapshot.eviction_steps
    ]

    losses = model.loss(snapshots)
    losses["eviction"].backward()

    assert model.snapshot_history_tokens == expected_histories, (
        "loss replay should consume the same micro-step history prefixes captured at collect() time"
    )

    history_grad = 0.0
    for name, param in model.named_parameters():
        if "history_lstm" in name and param.grad is not None:
            history_grad += float(param.grad.abs().sum().item())

    assert history_grad > 0.0, "history_lstm should receive gradients through collect()->loss() replay"


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
    assert snapshot is not None, "expected training eviction snapshot for target sequence"

    alg.access(target)

    runtime_lengths = list(runtime_model.runtime_history_lengths)
    snapshot_lengths = [len(step.history_tokens) for step in snapshot.eviction_steps]

    assert runtime_lengths == snapshot_lengths == [4, 5], (
        "inference-time history visibility and training snapshot prefixes should align step by step"
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
    losses["eviction"].backward()

    for history_tokens in model.snapshot_history_tokens:
        assert len(history_tokens) <= model.max_attention_history, (
            "replayed history prefixes should respect the same bounded window as runtime memory"
        )


if __name__ == "__main__":
    test_step_i_sees_only_prior_prefix_and_step_i_plus_1_sees_h_i()
    test_collect_snapshot_history_tokens_match_micro_step_prefixes()
    test_collect_to_loss_replays_same_micro_step_history_prefixes()
    test_protection_uses_current_prefix_not_future_suffix()
    test_train_infer_history_lengths_align_for_same_sequence()
    test_history_window_bounding_matches_runtime_and_replay()
    print("STEP-WISE HISTORY UPDATE TESTS PASSED")
