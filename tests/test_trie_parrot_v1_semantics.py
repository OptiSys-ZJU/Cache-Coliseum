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


def test_collect_buffers_microstep_access_snapshots():
    cache = TrieTrainingCache(max_node_num=4, model=None)
    sequences = [
        [9],
        [1, 2, 3],
    ]
    cache.load_future_accesses(sequences)
    cache.set_model_prob(0.0)

    cache.collect(sequences[0])
    returned_snapshot, _ = cache.collect(sequences[1])

    assert returned_snapshot is not None
    returned_paths = [
        step.current_path for step in returned_snapshot.eviction_steps
    ]
    assert returned_paths == [(1,), (1, 2), (1, 2, 3)]
    assert all(
        step.step_kind == "microstep_access"
        for step in returned_snapshot.eviction_steps
    )
    assert cache.alg.eviction_count == 0

    buffered = cache.get_snapshots()
    assert len(buffered) == 1
    buffered_paths = [
        step.current_path for snapshot in buffered for step in snapshot.eviction_steps
    ]
    assert buffered_paths == [(1,), (1, 2), (1, 2, 3)]


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
    assert first_snapshot is None, "empty first microstep has no candidates"

    second_snapshot, _ = cache.collect(sequences[1])
    assert second_snapshot is not None
    first_step, second_step = second_snapshot.eviction_steps

    assert first_step.step_kind == "microstep_access"
    assert [tuple(path) for path in first_step.leaf_paths] == [(1,)]
    assert tuple(first_step.current_path) == (1,)
    assert tuple(first_step.microstep_history_paths) == ((1,), (1,))
    assert tuple(second_step.current_path) == (1, 2)
    assert tuple(second_step.microstep_history_paths) == ((1,), (1,), (1, 2))
    assert list(cache.alg.microstep_history_path_window) == [(1,), (1,), (1, 2)]


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
    assert [step.step_kind for step in snapshot.eviction_steps] == [
        "microstep_access",
        "microstep_access",
    ]
    assert [tuple(step.current_path) for step in snapshot.eviction_steps] == [
        (1,),
        (1, 2),
    ]
    assert [tuple(path) for path in snapshot.eviction_steps[0].leaf_paths] == [(1, 2)]
    assert [tuple(path) for path in snapshot.eviction_steps[1].leaf_paths] == [(1, 2)]
    assert tuple(snapshot.eviction_steps[0].microstep_history_paths) == (
        (1,),
        (1, 2),
        (1,),
    )
    assert tuple(snapshot.eviction_steps[1].microstep_history_paths) == (
        (1,),
        (1, 2),
        (1,),
        (1, 2),
    )


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


def test_microstep_hit_line_participates_with_zero_distance():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    model.eval()
    cache = TrieTrainingCache(max_node_num=10, model=model)
    sequences = [
        [1, 2, 3],
        [4],
        [1, 2, 3],
    ]
    cache.load_future_accesses(sequences)

    cache.collect(sequences[0])
    cache.collect(sequences[1])
    snapshot, hit = cache.collect(sequences[2])

    assert hit
    assert snapshot is not None
    for step in snapshot.eviction_steps:
        distances_by_path = {
            tuple(path): distance
            for path, distance in zip(step.leaf_paths, step.oracle_distances)
        }
        assert (1, 2, 3) in distances_by_path
        assert distances_by_path[(1, 2, 3)] == 0


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


def test_actual_eviction_candidates_exclude_protected_current_path():
    cache = TrieTrainingCache(max_node_num=10, model=None)
    for seq in ([1, 2], [3], [4]):
        _insert_raw(cache.alg, seq)

    snapshot = cache._evict_and_collect(
        evict_num=1,
        this_node=cache.alg.root_node,
        current_path=[1, 2],
        step_index=1,
    )

    paths = [tuple(path) for path in snapshot.eviction_steps[0].leaf_paths]
    assert (1, 2) not in paths
    assert (3,) in paths
    assert (4,) in paths


def test_eviction_does_not_append_history():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    cache = TrieTrainingCache(max_node_num=10, model=model)
    for seq in ([1, 2], [3], [4]):
        _insert_raw(cache.alg, seq)

    node = _node_for_path(cache.alg, [3])
    cache.alg._record_microstep_history_path([3], node.hidden_state)
    before_paths = tuple(cache.alg.microstep_history_path_window)
    before_len = len(cache.alg.microstep_history_hidden_states)

    cache._evict_and_collect(
        evict_num=1,
        this_node=cache.alg.root_node,
        current_path=[8],
        step_index=0,
    )

    assert tuple(cache.alg.microstep_history_path_window) == before_paths
    assert len(cache.alg.microstep_history_hidden_states) == before_len


def test_request_touch_appends_microstep_history_slots():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    alg = TrieModelPredictAlgorithm(max_node_num=10, model=model)

    alg.access([1, 2, 3])

    assert list(alg.microstep_history_path_window) == [(1,), (1, 2), (1, 2, 3)]
    assert len(alg.microstep_history_hidden_states) == 3
    assert list(alg.request_history_path_window) == [(1, 2, 3)]
    assert len(alg.request_history_hidden_states) == 1


def test_candidate_is_query_only_not_direct_scorer_input():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=8)
    model.eval()

    microstep_history_memory = torch.randn(1, model.hidden_size)
    request_history_memory = torch.randn(1, model.hidden_size)
    candidate_states = [
        torch.randn(1, model.hidden_size),
        torch.randn(1, model.hidden_size),
    ]
    lru_features = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
    ]

    with torch.no_grad():
        logits, pred_reuse = model.forward(
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=candidate_states,
            inference=True,
        )

    assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)
    assert torch.allclose(pred_reuse[:, 0], pred_reuse[:, 1], atol=1e-6)


def test_lru_prior_directly_conditions_score():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=1.5,
        lru_prior_alpha_max=1.5,
        lru_prior_alpha_learnable=False,
    )
    model.eval()
    with torch.no_grad():
        model.request_head.weight.zero_()
        model.request_head.bias.zero_()
        model.micro_head.weight.zero_()
        model.micro_head.bias.zero_()

    microstep_history_memory = torch.randn(1, model.hidden_size)
    request_history_memory = torch.randn(1, model.hidden_size)
    candidate_states = [
        torch.randn(1, model.hidden_size),
        torch.randn(1, model.hidden_size),
    ]
    lru_features = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (8.0, 8.0, 8.0, 8.0, 1.0),
    ]

    with torch.no_grad():
        logits, pred_reuse = model.forward(
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=candidate_states,
            inference=True,
        )

    expected_gap = 1.5 * (math.log1p(8.0) - math.log1p(1.0))
    assert logits[0, 1] > logits[0, 0]
    assert torch.allclose(logits[0, 1] - logits[0, 0], torch.tensor(expected_gap), atol=1e-6)
    assert pred_reuse.shape == logits.shape


def test_lru_prior_alpha_initialization_and_nonnegative():
    learnable = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=0.25,
        lru_prior_alpha_min=0.25,
        lru_prior_alpha_max=1.5,
    )
    assert learnable.lru_prior_alpha().item() >= 0.25
    assert math.isclose(learnable.lru_prior_alpha().item(), 0.25, rel_tol=0.0, abs_tol=1e-6)

    with torch.no_grad():
        learnable.lru_prior_raw_alpha.fill_(-100.0)
    assert learnable.lru_prior_alpha().item() >= 0.25

    with torch.no_grad():
        learnable.lru_prior_raw_alpha.fill_(100.0)
    assert learnable.lru_prior_alpha().item() >= 0.25
    assert learnable.lru_prior_alpha().item() <= 1.5

    fixed = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=0.1,
        lru_prior_alpha_min=0.25,
        lru_prior_alpha_max=1.5,
        lru_prior_alpha_learnable=False,
    )
    assert math.isclose(fixed.lru_prior_alpha().item(), 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert "lru_prior_raw_alpha" not in dict(fixed.named_parameters())

    fixed_high = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=2.5,
        lru_prior_alpha_min=0.25,
        lru_prior_alpha_max=1.5,
        lru_prior_alpha_learnable=False,
    )
    assert math.isclose(fixed_high.lru_prior_alpha().item(), 1.5, rel_tol=0.0, abs_tol=1e-6)


def test_old_lru_prior_raw_alpha_checkpoint_preserves_alpha_with_min_bound():
    old_model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=0.75,
        lru_prior_alpha_max=1.5,
    )
    old_alpha = old_model.lru_prior_alpha().detach().item()
    old_state = {
        key: value.clone()
        for key, value in old_model.state_dict().items()
        if key != "lru_prior_alpha_encoding_version"
    }

    new_model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=0.25,
        lru_prior_alpha_min=0.25,
        lru_prior_alpha_max=1.5,
    )
    migration = new_model.load_state_dict_compatible(old_state)

    assert migration["migrated"]
    assert math.isclose(
        new_model.lru_prior_alpha().detach().item(),
        old_alpha,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_lcp_feature_head_affects_forward_when_enabled():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=0.0,
        lru_prior_alpha_learnable=False,
        use_lcp_features=True,
    )
    model.eval()
    with torch.no_grad():
        model.request_head.weight.zero_()
        model.request_head.bias.zero_()
        model.micro_head.weight.zero_()
        model.micro_head.bias.zero_()
        for parameter in model.lcp_head.parameters():
            parameter.zero_()
        model.lcp_head[0].weight[0, 2] = 1.0
        model.lcp_head[2].weight[0, 0] = 1.0

    microstep_history_memory = torch.zeros(1, model.hidden_size)
    request_history_memory = torch.zeros(1, model.hidden_size)
    candidate_states = [
        torch.zeros(1, model.hidden_size),
        torch.zeros(1, model.hidden_size),
    ]
    lru_features = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
    ]
    lcp_features = [
        (0.0, 0.0, 0.1, 0.0, 1.0),
        (0.0, 0.0, 0.9, 0.0, 1.0),
    ]

    with torch.no_grad():
        logits, _ = model.forward(
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=candidate_states,
            lcp_features=lcp_features,
            inference=True,
        )

    assert logits[0, 1] > logits[0, 0]
    assert torch.allclose(logits[0, 1] - logits[0, 0], torch.tensor(0.8), atol=1e-6)


def test_lru_prior_forward_and_batch_paths_match():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=8,
        lru_prior_alpha_init=1.25,
        lru_prior_alpha_learnable=False,
    )
    model.eval()
    with torch.no_grad():
        model.request_head.weight.zero_()
        model.request_head.bias.fill_(0.75)
        model.micro_head.weight.zero_()
        model.micro_head.bias.fill_(-0.25)
        model.score_mix_logits.zero_()

    candidate_paths = [(1,), (2, 3), (4,)]
    lru_features = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (4.0, 4.0, 4.0, 4.0, 2.0),
        (9.0, 9.0, 9.0, 9.0, 1.0),
    ]

    with torch.no_grad():
        forward_logits, forward_reuse = model.forward(
            torch.empty(0, model.hidden_size),
            torch.empty(0, model.hidden_size),
            lru_features,
            candidate_paths=candidate_paths,
            inference=False,
        )
        batch_logits, batch_reuse, batch_mask = model.forward_batched(
            [[]],
            [candidate_paths],
            [[]],
            [lru_features],
        )

    assert torch.equal(batch_mask, torch.ones_like(batch_mask))
    assert torch.allclose(batch_logits, forward_logits, atol=1e-6)
    assert torch.allclose(batch_reuse, forward_reuse, atol=1e-6)


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
        microstep_history_paths=((9,), (9, 8)),
        request_history_paths=((7,),),
        lru_features=(
            (1.0, 1.0, 1.0, 1.0, 2.0),
            (2.0, 2.0, 2.0, 2.0, 1.0),
            (3.0, 3.0, 3.0, 3.0, 2.0),
        ),
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
    test_collect_buffers_microstep_access_snapshots()
    test_request_snapshot_uses_pre_access_live_leaf_set()
    test_hit_without_eviction_still_collects_request_snapshot()
    test_pre_access_snapshot_labels_current_hit_as_immediate_reuse()
    test_microstep_hit_line_participates_with_zero_distance()
    test_continuous_eviction_adds_parent_candidate_and_recomputes_distances()
    test_actual_eviction_candidates_exclude_protected_current_path()
    test_eviction_does_not_append_history()
    test_request_touch_appends_microstep_history_slots()
    test_candidate_is_query_only_not_direct_scorer_input()
    test_lru_prior_directly_conditions_score()
    test_lru_prior_alpha_initialization_and_nonnegative()
    test_old_lru_prior_raw_alpha_checkpoint_preserves_alpha_with_min_bound()
    test_lcp_feature_head_affects_forward_when_enabled()
    test_lru_prior_forward_and_batch_paths_match()
    test_loss_handles_finite_and_inf_oracle_distances_without_nan()
    print("TRIE-PARROT V1 SEMANTICS TESTS PASSED")
