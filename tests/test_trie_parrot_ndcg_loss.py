#!/usr/bin/env python3
"""Tests for Trie-PARROT NDCG/ranking loss semantics."""
import math
import os
import sys
import json
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from model.trie_model.model import TrieParrotModel


def make_model(**kwargs):
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=32,
        **kwargs,
    )
    model.train()
    return model


def make_snapshot(oracle_distances):
    step = SimpleNamespace(
        leaf_paths=[(1,), (2,), (3,)],
        oracle_distances=oracle_distances,
        oracle_target=max(range(len(oracle_distances)), key=lambda idx: oracle_distances[idx]),
        history_paths=((9,), (9, 8)),
        num_candidates=3,
    )
    return SimpleNamespace(eviction_steps=[step])


def stepwise_reference_loss(
    model,
    snapshots,
    max_candidates=None,
    max_steps_per_snapshot=None,
):
    """Previous per-step loss implementation, kept as a test oracle."""
    device = next(model.parameters()).device
    ranking_losses = []
    reuse_losses = []
    ce_losses = []

    for snapshot in snapshots:
        eviction_steps = snapshot.eviction_steps
        if (
            max_steps_per_snapshot is not None
            and len(eviction_steps) > max_steps_per_snapshot
        ):
            quota = max(1, max_steps_per_snapshot)
            stride = len(eviction_steps) / quota
            eviction_steps = [
                eviction_steps[min(int(slot * stride), len(eviction_steps) - 1)]
                for slot in range(quota)
            ]

        for step in eviction_steps:
            if step.num_candidates < 2:
                continue

            history_paths = getattr(step, "history_paths", None)
            if history_paths is not None:
                history_memory = model._encode_history_paths(history_paths, device)
            else:
                if hasattr(step, "history_tokens"):
                    raise ValueError(
                        "Trie-PARROT v1 snapshots must provide history_paths; "
                        "history_tokens is a legacy prefix/token-history format"
                    )
                history_memory = None

            selected_indices, target_idx = model._candidate_subset(
                step.num_candidates,
                step.oracle_target,
                max_candidates,
                getattr(step, "required_candidate_indices", None),
            )
            candidate_paths = [step.leaf_paths[idx] for idx in selected_indices]
            logits, pred_log_reuse = model(
                history_memory,
                candidate_paths=candidate_paths,
                inference=False,
            )

            oracle_distances = getattr(step, "oracle_distances", None)
            if oracle_distances is not None:
                relevances = model._transform_oracle_distances(
                    oracle_distances,
                    selected_indices,
                    device,
                ).unsqueeze(0)
                ranking_losses.append(model._approx_ndcg_loss(logits, relevances))

                if model.reuse_loss_weight > 0:
                    reuse_losses.append(torch.nn.functional.mse_loss(
                        pred_log_reuse,
                        relevances,
                    ).unsqueeze(0))

            if model.ce_loss_weight > 0:
                target_distribution = model._ce_target_distribution(
                    oracle_distances,
                    selected_indices,
                    target_idx,
                    device,
                )
                ce_losses.append(
                    model._distribution_cross_entropy(
                        logits,
                        target_distribution,
                    ).unsqueeze(0)
                )

    losses = {}
    if ranking_losses:
        losses["ranking"] = model.ranking_loss_weight * torch.cat(ranking_losses).mean()
    else:
        losses["ranking"] = torch.tensor(0.0, device=device, requires_grad=True)
    if reuse_losses:
        losses["reuse"] = model.reuse_loss_weight * torch.cat(reuse_losses).mean()
    else:
        losses["reuse"] = torch.tensor(0.0, device=device, requires_grad=True)
    if ce_losses:
        losses["ce"] = model.ce_loss_weight * torch.cat(ce_losses).mean()
    else:
        losses["ce"] = torch.tensor(0.0, device=device, requires_grad=True)
    return losses


def test_relevance_transform_parrot_style():
    model = make_model()
    distances = [0.0, 1.0, 2.0, 10.0, float("inf")]
    transformed = model._transform_oracle_distances(
        distances,
        list(range(len(distances))),
        torch.device("cpu"),
    )

    expected = torch.tensor(
        [0.0, 0.0, math.log10(2.0), 1.0, 5.0],
        dtype=torch.float32,
    )
    assert torch.allclose(transformed.cpu(), expected, atol=1e-6)
    assert transformed[2] > transformed[1]
    assert transformed[3] > transformed[2]
    assert transformed[4] > transformed[3]


def test_ndcg_position_sign_improving_high_relevance_score_lowers_loss():
    model = make_model()
    relevance = torch.tensor([[0.0, 5.0]], dtype=torch.float32)

    low_score_for_best = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    high_score_for_best = torch.tensor([[0.0, 4.0]], dtype=torch.float32)

    bad_loss = model._approx_ndcg_loss(low_score_for_best, relevance)
    good_loss = model._approx_ndcg_loss(high_score_for_best, relevance)

    assert good_loss.item() < bad_loss.item(), (
        "raising the score of the high-relevance candidate must lower ranking loss"
    )


def test_ranking_loss_prefers_oracle_order():
    model = make_model()
    relevance = torch.tensor([[0.0, 1.0, 5.0]], dtype=torch.float32)

    correct = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    reversed_scores = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)

    correct_loss = model._approx_ndcg_loss(correct, relevance)
    reversed_loss = model._approx_ndcg_loss(reversed_scores, relevance)

    assert correct_loss.item() < reversed_loss.item()


def test_ndcg_gain_matches_parrot_expm1_relevance():
    model = make_model()
    scores = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    relevance = torch.tensor([[1.0, 5.0]], dtype=torch.float32)

    positions = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
    gains = torch.expm1(relevance)
    dcg = (gains / torch.log2(positions + 1.0)).sum(dim=-1)
    ideal_positions = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    idcg = (torch.sort(gains, dim=-1, descending=True).values / torch.log2(
        ideal_positions + 1.0
    )).sum(dim=-1)
    expected = -(dcg / idcg)

    actual = model._approx_ndcg_loss(scores, relevance)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_ranking_loss_uses_all_candidates_not_only_oracle_target():
    model = make_model(reuse_loss_weight=0.0)
    scores = torch.tensor([[0.2, 0.1, 0.9]], dtype=torch.float32)

    # Candidate 2 remains the oracle target in both cases, but candidate 1's
    # relevance changes. A full-candidate ranking loss must notice this.
    rel_a = model._transform_oracle_distances(
        [1.0, 2.0, float("inf")],
        [0, 1, 2],
        torch.device("cpu"),
    ).unsqueeze(0)
    rel_b = model._transform_oracle_distances(
        [1.0, 100.0, float("inf")],
        [0, 1, 2],
        torch.device("cpu"),
    ).unsqueeze(0)

    loss_a = model._approx_ndcg_loss(scores, rel_a)
    loss_b = model._approx_ndcg_loss(scores, rel_b)

    assert not torch.allclose(loss_a, loss_b), (
        "ranking loss should change when a non-target candidate's relevance changes"
    )


def test_loss_uses_all_candidates_by_default():
    model = make_model(reuse_loss_weight=0.0)
    num_candidates = 20
    step = SimpleNamespace(
        leaf_paths=[(idx + 1,) for idx in range(num_candidates)],
        oracle_distances=[float(idx + 1) for idx in range(num_candidates - 1)]
        + [float("inf")],
        oracle_target=num_candidates - 1,
        history_paths=((9,),),
        num_candidates=num_candidates,
    )
    snapshot = SimpleNamespace(eviction_steps=[step])
    seen_candidate_counts = []
    original_forward = model.forward_batched

    def spy_forward_batched(history_paths_batch, candidate_paths_batch):
        seen_candidate_counts.extend(
            len(candidate_paths)
            for candidate_paths in candidate_paths_batch
        )
        return original_forward(history_paths_batch, candidate_paths_batch)

    model.forward_batched = spy_forward_batched
    model.loss([snapshot])

    assert seen_candidate_counts == [num_candidates]
    assert model.last_loss_stats["full_steps"] == 1
    assert model.last_loss_stats["capped_steps"] == 0
    assert model.last_loss_stats["candidate_count"] == num_candidates


def test_loss_candidate_cap_keeps_current_hit_required_candidate():
    model = make_model(reuse_loss_weight=0.0)
    step = SimpleNamespace(
        leaf_paths=[(idx + 1,) for idx in range(6)],
        oracle_distances=[0.0, 2.0, 3.0, 4.0, 5.0, float("inf")],
        oracle_target=5,
        required_candidate_indices=(0,),
        history_paths=((9,),),
        num_candidates=6,
    )
    snapshot = SimpleNamespace(eviction_steps=[step])
    seen_candidate_paths = []
    original_forward = model.forward_batched

    def spy_forward_batched(history_paths_batch, candidate_paths_batch):
        seen_candidate_paths.extend(
            tuple(candidate_paths)
            for candidate_paths in candidate_paths_batch
        )
        return original_forward(history_paths_batch, candidate_paths_batch)

    model.forward_batched = spy_forward_batched
    model.loss([snapshot], max_candidates=2)

    assert seen_candidate_paths == [((1,), (6,))]
    assert model.last_loss_stats["full_steps"] == 0
    assert model.last_loss_stats["capped_steps"] == 1
    assert model.last_loss_stats["candidate_count"] == 2


def test_batched_loss_matches_stepwise_reference_with_padding():
    snapshots = [
        SimpleNamespace(eviction_steps=[
            SimpleNamespace(
                leaf_paths=[(1,), (2, 3), (4,)],
                oracle_distances=[1.0, 5.0, float("inf")],
                oracle_target=2,
                required_candidate_indices=(0,),
                history_paths=((9,),),
                num_candidates=3,
            ),
            SimpleNamespace(
                leaf_paths=[(5,), (6, 7)],
                oracle_distances=[2.0, float("inf")],
                oracle_target=1,
                history_paths=((9,), (9, 8)),
                num_candidates=2,
            ),
        ]),
        SimpleNamespace(eviction_steps=[
            SimpleNamespace(
                leaf_paths=[(10,), (11,), (12,), (13, 14)],
                oracle_distances=[0.0, 3.0, float("inf"), 6.0],
                oracle_target=2,
                history_paths=(),
                num_candidates=4,
            ),
            SimpleNamespace(
                leaf_paths=[(15, 16), (17,), (18,)],
                oracle_distances=[float("inf"), 4.0, 4.0],
                oracle_target=0,
                history_paths=((10,),),
                num_candidates=3,
            ),
        ]),
    ]

    for candidate_scorer_mode in ("history_only", "candidate_history_concat"):
        model = make_model(
            reuse_loss_weight=0.2,
            ce_loss_weight=0.5,
            ce_target_policy="top_set",
            candidate_scorer_mode=candidate_scorer_mode,
        )
        expected = stepwise_reference_loss(model, snapshots, max_candidates=3)
        actual = model.loss(snapshots, max_candidates=3)

        for name in ("ranking", "reuse", "ce"):
            assert torch.allclose(actual[name], expected[name], atol=1e-6), (
                candidate_scorer_mode,
                name,
            )


def test_loss_batches_same_time_steps_across_windows():
    model = make_model(reuse_loss_weight=0.0)
    snapshots = [
        SimpleNamespace(eviction_steps=[
            SimpleNamespace(
                leaf_paths=[(1,), (2,)],
                oracle_distances=[1.0, float("inf")],
                oracle_target=1,
                history_paths=((9,),),
                num_candidates=2,
            ),
            SimpleNamespace(
                leaf_paths=[(3,), (4,), (5,)],
                oracle_distances=[1.0, 2.0, float("inf")],
                oracle_target=2,
                history_paths=((9,), (10,)),
                num_candidates=3,
            ),
        ]),
        SimpleNamespace(eviction_steps=[
            SimpleNamespace(
                leaf_paths=[(6,), (7,), (8,)],
                oracle_distances=[1.0, 2.0, float("inf")],
                oracle_target=2,
                history_paths=((11,),),
                num_candidates=3,
            ),
            SimpleNamespace(
                leaf_paths=[(9,), (10,)],
                oracle_distances=[float("inf"), 1.0],
                oracle_target=0,
                history_paths=((11,), (12,)),
                num_candidates=2,
            ),
        ]),
    ]
    seen_batches = []
    original_forward = model.forward_batched

    def spy_forward_batched(history_paths_batch, candidate_paths_batch):
        seen_batches.append((
            len(history_paths_batch),
            tuple(len(paths) for paths in candidate_paths_batch),
        ))
        return original_forward(history_paths_batch, candidate_paths_batch)

    model.forward_batched = spy_forward_batched
    model.loss(snapshots)

    assert seen_batches == [(2, (2, 3)), (2, (3, 2))]


def test_loss_has_finite_reuse_with_inf_distance():
    model = make_model(reuse_loss_weight=0.1)
    losses = model.loss([make_snapshot([1.0, 10.0, float("inf")])])
    total = sum(losses.values())

    assert torch.isfinite(losses["ranking"])
    assert torch.isfinite(losses["reuse"])
    assert torch.isfinite(total)
    total.backward()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_ce_optional_default_zero_and_enabled_path():
    default_model = make_model()
    default_losses = default_model.loss([make_snapshot([1.0, 10.0, float("inf")])])
    assert default_model.ce_loss_weight == 0.0
    assert default_losses["ce"].item() == 0.0

    ce_model = make_model(ce_loss_weight=0.5)
    ce_losses = ce_model.loss([make_snapshot([1.0, 10.0, float("inf")])])
    assert ce_losses["ce"].item() > 0.0


def test_argmax_ce_matches_single_target_cross_entropy():
    model = make_model(ce_loss_weight=1.0)
    logits = torch.tensor([[0.0, 2.0, 4.0]], dtype=torch.float32)
    target_distribution = model._ce_target_distribution(
        [1.0, 10.0, float("inf")],
        [0, 1, 2],
        2,
        torch.device("cpu"),
    )
    actual = model._distribution_cross_entropy(logits, target_distribution)
    expected = torch.nn.functional.cross_entropy(
        logits,
        torch.tensor([2], dtype=torch.long),
    )

    assert torch.allclose(target_distribution, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(actual, expected)


def test_top_set_ce_targets_all_max_relevance_candidates():
    model = make_model(ce_loss_weight=1.0, ce_target_policy="top_set")
    logits = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
    target_distribution = model._ce_target_distribution(
        [1.0, float("inf"), float("inf")],
        [0, 1, 2],
        1,
        torch.device("cpu"),
    )
    actual = model._distribution_cross_entropy(logits, target_distribution)
    expected = -0.5 * (
        torch.nn.functional.log_softmax(logits, dim=-1)[0, 1]
        + torch.nn.functional.log_softmax(logits, dim=-1)[0, 2]
    )

    assert torch.allclose(target_distribution, torch.tensor([[0.0, 0.5, 0.5]]))
    assert torch.allclose(actual, expected)


def test_top_set_ce_respects_selected_subset():
    model = make_model(ce_loss_weight=1.0, ce_target_policy="top_set")
    logits = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    target_distribution = model._ce_target_distribution(
        [float("inf"), float("inf"), 1.0],
        [0, 2],
        0,
        torch.device("cpu"),
    )
    actual = model._distribution_cross_entropy(logits, target_distribution)
    expected = -torch.nn.functional.log_softmax(logits, dim=-1)[0, 0]

    assert torch.allclose(target_distribution, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(actual, expected)


def test_invalid_ce_target_policy_rejected():
    try:
        make_model(ce_target_policy="bogus")
    except ValueError as exc:
        assert "ce_target_policy" in str(exc)
    else:
        raise AssertionError("invalid ce_target_policy should fail fast")


def test_invalid_candidate_scorer_mode_rejected():
    try:
        make_model(candidate_scorer_mode="bogus")
    except ValueError as exc:
        assert "candidate_scorer_mode" in str(exc)
    else:
        raise AssertionError("invalid candidate_scorer_mode should fail fast")


def test_legacy_history_tokens_snapshot_rejected():
    model = make_model()
    step = SimpleNamespace(
        leaf_paths=[(1,), (2,)],
        oracle_distances=[1.0, float("inf")],
        oracle_target=1,
        history_tokens=(9, 8),
        num_candidates=2,
    )
    snapshot = SimpleNamespace(eviction_steps=[step])

    try:
        model.loss([snapshot])
    except ValueError as exc:
        assert "history_paths" in str(exc)
        assert "history_tokens" in str(exc)
    else:
        raise AssertionError("legacy history_tokens snapshots should fail fast")


def test_empty_history_paths_are_not_real_history_slots():
    model = make_model()
    device = torch.device("cpu")

    assert model._encode_history_paths(None, device) is None
    assert model._encode_history_paths((), device) is None

    memory, has_history = model._prepare_history_memory(None, device)
    assert memory.shape == (1, model.hidden_size)
    assert not has_history


def test_from_config_reads_candidate_scorer_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "vocab_size": 128,
                    "node_embed_dim": 16,
                    "history_embed_dim": 16,
                    "hidden_size": 32,
                    "candidate_scorer_mode": "candidate_history_concat",
                    "ce_target_policy": "top_set",
                },
                f,
            )

        model = TrieParrotModel.from_config(config_path)
        assert model.candidate_scorer_mode == "candidate_history_concat"
        assert model.ce_target_policy == "top_set"


if __name__ == "__main__":
    test_relevance_transform_parrot_style()
    test_ndcg_position_sign_improving_high_relevance_score_lowers_loss()
    test_ranking_loss_prefers_oracle_order()
    test_ndcg_gain_matches_parrot_expm1_relevance()
    test_ranking_loss_uses_all_candidates_not_only_oracle_target()
    test_loss_uses_all_candidates_by_default()
    test_loss_candidate_cap_keeps_current_hit_required_candidate()
    test_batched_loss_matches_stepwise_reference_with_padding()
    test_loss_batches_same_time_steps_across_windows()
    test_loss_has_finite_reuse_with_inf_distance()
    test_ce_optional_default_zero_and_enabled_path()
    test_argmax_ce_matches_single_target_cross_entropy()
    test_top_set_ce_targets_all_max_relevance_candidates()
    test_top_set_ce_respects_selected_subset()
    test_invalid_ce_target_policy_rejected()
    test_invalid_candidate_scorer_mode_rejected()
    test_legacy_history_tokens_snapshot_rejected()
    test_empty_history_paths_are_not_real_history_slots()
    test_from_config_reads_candidate_scorer_mode()
    print("TRIE-PARROT NDCG LOSS TESTS PASSED")
