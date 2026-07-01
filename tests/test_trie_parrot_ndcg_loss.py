#!/usr/bin/env python3
"""Tests for lru-trie Trie-PARROT ranking/loss semantics."""
import json
import math
import os
import sys
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


def lru_features_for(leaf_paths):
    rows = []
    for idx, path in enumerate(leaf_paths):
        age = float(idx + 1)
        rows.append((age, age, age, age, float(len(path))))
    return tuple(rows)


def make_step(
    leaf_paths,
    oracle_distances,
    microstep_history_paths=((9,), (9, 8)),
    request_history_paths=((7,),),
    required_candidate_indices=None,
    step_kind="microstep_access",
    lru_target=None,
    oracle_top_set=None,
    lcp_diagnostics=None,
):
    resolved_oracle_target = max(
        range(len(oracle_distances)),
        key=lambda idx: oracle_distances[idx],
    )
    if oracle_top_set is None:
        max_distance = max(oracle_distances)
        oracle_top_set = tuple(
            idx for idx, distance in enumerate(oracle_distances)
            if distance == max_distance
        )
    return SimpleNamespace(
        step_kind=step_kind,
        leaf_paths=list(leaf_paths),
        oracle_distances=list(oracle_distances),
        oracle_target=resolved_oracle_target,
        oracle_top_set=tuple(oracle_top_set),
        lru_target=lru_target,
        lcp_diagnostics=tuple(lcp_diagnostics or ()),
        required_candidate_indices=required_candidate_indices,
        microstep_history_paths=tuple(microstep_history_paths),
        request_history_paths=tuple(request_history_paths),
        lru_features=lru_features_for(leaf_paths),
        num_candidates=len(leaf_paths),
    )


def make_snapshot(oracle_distances):
    return SimpleNamespace(
        eviction_steps=[
            make_step([(1,), (2,), (3,)], oracle_distances),
        ],
    )


def stepwise_reference_loss(
    model,
    snapshots,
    max_candidates=None,
    max_steps_per_snapshot=None,
    warmup_steps_per_snapshot=0,
):
    """Previous per-step loss shape, kept as a batching oracle."""
    device = next(model.parameters()).device
    ranking_losses = []
    reuse_losses = []
    ce_losses = []

    for snapshot in snapshots:
        eviction_steps = snapshot.eviction_steps
        warmup_steps = max(0, int(warmup_steps_per_snapshot or 0))
        if warmup_steps > 0:
            eviction_steps = eviction_steps[warmup_steps:]
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

            selected_indices, target_idx = model._candidate_subset(
                step.num_candidates,
                step.oracle_target,
                max_candidates,
                getattr(step, "required_candidate_indices", None),
            )
            candidate_paths = [step.leaf_paths[idx] for idx in selected_indices]
            selected_lru_features = [
                step.lru_features[idx] for idx in selected_indices
            ]
            micro_memory = model._encode_history_paths(
                step.microstep_history_paths,
                device,
                max_history=model.max_microstep_history,
            )
            request_memory = model._encode_request_history_paths(
                step.request_history_paths,
                device,
            )
            logits, pred_log_reuse = model(
                micro_memory,
                request_memory,
                selected_lru_features,
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


def test_loss_warmup_matches_parrot_suffix_loss():
    model = make_model(reuse_loss_weight=0.2)
    snapshots = [
        SimpleNamespace(
            eviction_steps=[
                make_step([(1,), (2,), (3,)], [1, 2, float("inf")]),
                make_step([(1,), (2,), (3,)], [float("inf"), 1, 2]),
                make_step([(1,), (2,), (3,)], [1, float("inf"), 2]),
                make_step([(1,), (2,), (3,)], [2, 1, float("inf")]),
            ],
        ),
    ]

    batched = model.loss(snapshots, warmup_steps_per_snapshot=2)
    reference = stepwise_reference_loss(
        model,
        snapshots,
        warmup_steps_per_snapshot=2,
    )

    for name in ("ranking", "reuse", "ce"):
        assert torch.allclose(batched[name], reference[name], atol=1e-6), (
            name,
            batched[name].item(),
            reference[name].item(),
        )
    assert model.last_loss_stats["loss_steps"] == 2


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


def test_ndcg_position_sign_improving_high_relevance_score_lowers_loss():
    model = make_model()
    relevance = torch.tensor([[0.0, 5.0]], dtype=torch.float32)
    bad_loss = model._approx_ndcg_loss(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        relevance,
    )
    good_loss = model._approx_ndcg_loss(
        torch.tensor([[0.0, 4.0]], dtype=torch.float32),
        relevance,
    )
    assert good_loss.item() < bad_loss.item()


def test_loss_uses_all_candidates_by_default():
    model = make_model(reuse_loss_weight=0.0)
    num_candidates = 20
    step = make_step(
        [(idx + 1,) for idx in range(num_candidates)],
        [float(idx + 1) for idx in range(num_candidates - 1)] + [float("inf")],
    )
    snapshot = SimpleNamespace(eviction_steps=[step])
    seen_candidate_counts = []
    original_forward = model.forward_batched

    def spy_forward_batched(
        microstep_history_paths_batch,
        candidate_paths_batch,
        request_history_paths_batch,
        lru_features_batch,
    ):
        del microstep_history_paths_batch
        del request_history_paths_batch
        del lru_features_batch
        seen_candidate_counts.extend(
            len(candidate_paths)
            for candidate_paths in candidate_paths_batch
        )
        return original_forward(
            [step.microstep_history_paths],
            candidate_paths_batch,
            [step.request_history_paths],
            [step.lru_features],
        )

    model.forward_batched = spy_forward_batched
    model.loss([snapshot])

    assert seen_candidate_counts == [num_candidates]
    assert model.last_loss_stats["full_steps"] == 1
    assert model.last_loss_stats["capped_steps"] == 0
    assert model.last_loss_stats["candidate_count"] == num_candidates


def test_loss_candidate_cap_keeps_current_hit_required_candidate():
    model = make_model(reuse_loss_weight=0.0)
    step = make_step(
        [(idx + 1,) for idx in range(6)],
        [0.0, 2.0, 3.0, 4.0, 5.0, float("inf")],
        required_candidate_indices=(0,),
    )
    snapshot = SimpleNamespace(eviction_steps=[step])
    seen_candidate_paths = []
    original_forward = model.forward_batched

    def spy_forward_batched(
        microstep_history_paths_batch,
        candidate_paths_batch,
        request_history_paths_batch,
        lru_features_batch,
    ):
        seen_candidate_paths.extend(tuple(paths) for paths in candidate_paths_batch)
        return original_forward(
            microstep_history_paths_batch,
            candidate_paths_batch,
            request_history_paths_batch,
            lru_features_batch,
        )

    model.forward_batched = spy_forward_batched
    model.loss([snapshot], max_candidates=2)

    assert seen_candidate_paths == [((1,), (6,))]
    assert model.last_loss_stats["full_steps"] == 0
    assert model.last_loss_stats["capped_steps"] == 1
    assert model.last_loss_stats["candidate_count"] == 2


def test_loss_candidate_cap_forces_lru_and_oracle_top_set_candidates():
    model = make_model(reuse_loss_weight=0.0)
    step = make_step(
        [(idx + 1,) for idx in range(8)],
        [1.0, 2.0, float("inf"), 4.0, 5.0, 6.0, float("inf"), 0.0],
        required_candidate_indices=(7,),
        lru_target=0,
    )
    snapshot = SimpleNamespace(eviction_steps=[step])
    seen_candidate_paths = []
    original_forward = model.forward_batched

    def spy_forward_batched(
        microstep_history_paths_batch,
        candidate_paths_batch,
        request_history_paths_batch,
        lru_features_batch,
    ):
        seen_candidate_paths.extend(tuple(paths) for paths in candidate_paths_batch)
        return original_forward(
            microstep_history_paths_batch,
            candidate_paths_batch,
            request_history_paths_batch,
            lru_features_batch,
        )

    model.forward_batched = spy_forward_batched
    model.loss([snapshot], max_candidates=3)

    assert seen_candidate_paths == [((1,), (3,), (7,), (8,))]
    assert model.last_loss_stats["candidate_count"] == 4
    assert model.last_loss_stats["lru_target_kept_count"] == 1
    assert model.last_loss_stats["oracle_top_set_kept_count"] == 1


def test_batched_loss_matches_stepwise_reference_with_padding():
    snapshots = [
        SimpleNamespace(eviction_steps=[
            make_step(
                [(1,), (2, 3), (4,)],
                [1.0, 5.0, float("inf")],
                required_candidate_indices=(0,),
            ),
            make_step(
                [(5,), (6, 7)],
                [2.0, float("inf")],
                microstep_history_paths=((9,), (9, 8)),
            ),
        ]),
        SimpleNamespace(eviction_steps=[
            make_step(
                [(10,), (11,), (12,), (13, 14)],
                [0.0, 3.0, float("inf"), 6.0],
                microstep_history_paths=(),
                request_history_paths=(),
            ),
            make_step(
                [(15, 16), (17,), (18,)],
                [float("inf"), 4.0, 4.0],
                microstep_history_paths=((10,),),
            ),
        ]),
    ]

    model = make_model(
        reuse_loss_weight=0.2,
        ce_loss_weight=0.5,
        ce_target_policy="top_set",
    )
    expected = stepwise_reference_loss(model, snapshots, max_candidates=3)
    actual = model.loss(snapshots, max_candidates=3)

    for name in ("ranking", "reuse", "ce"):
        assert torch.allclose(actual[name], expected[name], atol=1e-6), name


def test_loss_sum_reduction_matches_mean_times_counts():
    snapshots = [
        SimpleNamespace(eviction_steps=[
            make_step([(1,), (2,), (3,)], [1.0, 5.0, float("inf")]),
            make_step([(4,), (5,)], [2.0, float("inf")]),
        ]),
        SimpleNamespace(eviction_steps=[
            make_step([(6,), (7,), (8,)], [float("inf"), 4.0, 4.0]),
        ]),
    ]
    model = make_model(
        reuse_loss_weight=0.2,
        ce_loss_weight=0.5,
        ce_target_policy="top_set",
    )

    mean_losses = model.loss(snapshots, max_candidates=3)
    mean_stats = dict(model.last_loss_stats)
    sum_losses = model.loss(snapshots, max_candidates=3, reduction="sum")
    sum_stats = dict(model.last_loss_stats)

    for name in ("ranking", "reuse", "ce"):
        count = sum_stats[f"{name}_count"]
        assert count == mean_stats[f"{name}_count"]
        assert count > 0
        assert torch.allclose(
            sum_losses[name] / count,
            mean_losses[name],
            atol=1e-6,
        ), name


def test_from_config_lru_prior_alpha_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        learnable_path = os.path.join(tmpdir, "learnable.json")
        with open(learnable_path, "w") as f:
            json.dump(
                {
                    "vocab_size": 128,
                    "node_embed_dim": 16,
                    "hidden_size": 32,
                    "lru_prior_alpha_init": 1.5,
                    "lru_prior_alpha_min": 0.25,
                    "lru_prior_alpha_learnable": True,
                    "use_lcp_features": True,
                    "lcp_wrong_margin_weight": 0.15,
                    "lcp_wrong_margin": 0.3,
                    "lcp_wrong_ratio_threshold": 0.6,
                },
                f,
            )
        learnable = TrieParrotModel.from_config(learnable_path)
        assert math.isclose(learnable.lru_prior_alpha().item(), 1.5, abs_tol=1e-6)
        assert learnable.lru_prior_alpha_min == 0.25
        assert learnable.use_lcp_features
        assert hasattr(learnable, "lcp_head")
        assert learnable.lcp_wrong_margin_weight == 0.15
        assert learnable.lcp_wrong_margin == 0.3
        assert learnable.lcp_wrong_ratio_threshold == 0.6
        assert "lru_prior_raw_alpha" in dict(learnable.named_parameters())

        fixed_path = os.path.join(tmpdir, "fixed.json")
        with open(fixed_path, "w") as f:
            json.dump(
                {
                    "vocab_size": 128,
                    "node_embed_dim": 16,
                    "hidden_size": 32,
                    "lru_prior_alpha_fixed": 0.1,
                    "lru_prior_alpha_min": 0.25,
                },
                f,
            )
        fixed = TrieParrotModel.from_config(fixed_path)
        assert math.isclose(fixed.lru_prior_alpha().item(), 0.25, abs_tol=1e-6)
        assert "lru_prior_raw_alpha" not in dict(fixed.named_parameters())


def test_old_lru_head_checkpoint_migrates_to_lru_prior():
    model = make_model(lru_prior_alpha_init=1.25)
    old_state = dict(model.state_dict())
    old_state["score_mix_logits"] = torch.tensor([0.4, -0.2, 1.5])
    old_state["lru_head.0.weight"] = torch.ones(5)
    old_state["lru_head.0.bias"] = torch.zeros(5)
    old_state.pop("lru_prior_raw_alpha")

    fresh = make_model(lru_prior_alpha_init=1.25)
    load_info = fresh.load_state_dict_compatible(old_state)

    assert load_info["migrated"]
    assert any(key.startswith("lru_head.") for key in load_info["dropped_keys"])
    assert fresh.score_mix_logits.shape == torch.Size([2])
    assert torch.allclose(
        fresh.score_mix_logits.detach(),
        torch.tensor([0.4, -0.2]),
    )
    assert math.isclose(fresh.lru_prior_alpha().item(), 1.25, abs_tol=1e-6)


def test_loss_batches_same_time_steps_across_windows():
    model = make_model(reuse_loss_weight=0.0)
    snapshots = [
        SimpleNamespace(eviction_steps=[
            make_step([(1,), (2,)], [1.0, float("inf")]),
            make_step([(3,), (4,), (5,)], [1.0, 2.0, float("inf")]),
        ]),
        SimpleNamespace(eviction_steps=[
            make_step([(6,), (7,), (8,)], [1.0, 2.0, float("inf")]),
            make_step([(9,), (10,)], [float("inf"), 1.0]),
        ]),
    ]
    seen_batches = []
    original_forward = model.forward_batched

    def spy_forward_batched(
        microstep_history_paths_batch,
        candidate_paths_batch,
        request_history_paths_batch,
        lru_features_batch,
    ):
        seen_batches.append((
            len(microstep_history_paths_batch),
            tuple(len(paths) for paths in candidate_paths_batch),
        ))
        return original_forward(
            microstep_history_paths_batch,
            candidate_paths_batch,
            request_history_paths_batch,
            lru_features_batch,
        )

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
    assert torch.allclose(actual, expected)


def test_top_set_ce_loss_uses_logsumexp_top_set_formula():
    model = make_model(
        reuse_loss_weight=0.0,
        ranking_loss_weight=0.0,
        top_set_ce_weight=1.0,
    )
    step = make_step(
        [(1,), (2,), (3,)],
        [1.0, float("inf"), float("inf")],
    )
    logits = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
    pred_reuse = torch.zeros_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)

    def fake_forward_batched(*args, **kwargs):
        del args, kwargs
        return logits, pred_reuse, mask

    model.forward_batched = fake_forward_batched
    losses = model.loss([SimpleNamespace(eviction_steps=[step])])
    expected = torch.logsumexp(logits, dim=-1) - torch.logsumexp(
        logits[:, [1, 2]],
        dim=-1,
    )

    assert torch.allclose(losses["top_set_ce"], expected.squeeze(0))
    assert model.last_loss_stats["top_set_acc"] == 0.0


def test_hard_lru_margin_only_active_when_lru_not_in_top_set():
    model = make_model(
        reuse_loss_weight=0.0,
        ranking_loss_weight=0.0,
        hard_lru_margin_weight=1.0,
        hard_lru_margin=0.2,
    )
    hard_step = make_step(
        [(1,), (2,), (3,)],
        [float("inf"), 1.0, 2.0],
        lru_target=1,
    )
    easy_step = make_step(
        [(4,), (5,), (6,)],
        [float("inf"), 1.0, 2.0],
        lru_target=0,
    )
    logits_by_call = [
        torch.tensor([[0.0, 1.5, 0.2]], dtype=torch.float32),
        torch.tensor([[0.0, 1.5, 0.2]], dtype=torch.float32),
    ]

    def fake_forward_batched(*args, **kwargs):
        del args, kwargs
        row_logits = logits_by_call.pop(0)
        return (
            row_logits,
            torch.zeros_like(row_logits),
            torch.ones_like(row_logits, dtype=torch.bool),
        )

    model.forward_batched = fake_forward_batched
    losses = model.loss([SimpleNamespace(eviction_steps=[hard_step, easy_step])])
    expected = torch.nn.functional.softplus(
        torch.tensor(1.5) - torch.tensor(0.0) + 0.2
    )

    assert torch.allclose(losses["hard_lru_margin"], expected)
    assert model.last_loss_stats["hard_lru_cases_count"] == 1
    assert model.last_loss_stats["hard_lru_margin_count"] == 1


def test_lcp_wrong_margin_only_active_for_high_lcp_wrong_target():
    model = make_model(
        reuse_loss_weight=0.0,
        ranking_loss_weight=0.0,
        lcp_wrong_margin_weight=1.0,
        lcp_wrong_margin=0.2,
        lcp_wrong_ratio_threshold=0.5,
    )

    def lcp_rows(wrong_ratio):
        return (
            {
                "lcp_len": 0,
                "lcp_ratio_candidate": 0.0,
                "lcp_ratio_current": 0.0,
                "candidate_suffix_len": 1,
                "current_suffix_len": 1,
            },
            {
                "lcp_len": 1,
                "lcp_ratio_candidate": 1.0,
                "lcp_ratio_current": wrong_ratio,
                "candidate_suffix_len": 0,
                "current_suffix_len": 1,
            },
            {
                "lcp_len": 0,
                "lcp_ratio_candidate": 0.0,
                "lcp_ratio_current": 0.0,
                "candidate_suffix_len": 1,
                "current_suffix_len": 1,
            },
        )

    high_wrong = make_step(
        [(1,), (2,), (3,)],
        [float("inf"), 1.0, 2.0],
        lcp_diagnostics=lcp_rows(0.7),
    )
    low_wrong = make_step(
        [(4,), (5,), (6,)],
        [float("inf"), 1.0, 2.0],
        lcp_diagnostics=lcp_rows(0.4),
    )
    top_correct = make_step(
        [(7,), (8,), (9,)],
        [float("inf"), 1.0, 2.0],
        lcp_diagnostics=lcp_rows(0.9),
    )
    logits = torch.tensor(
        [
            [0.0, 1.5, 0.2],
            [0.0, 1.5, 0.2],
            [1.5, 0.0, 0.2],
        ],
        dtype=torch.float32,
    )

    def fake_forward_batched(*args, **kwargs):
        del args, kwargs
        return (
            logits,
            torch.zeros_like(logits),
            torch.ones_like(logits, dtype=torch.bool),
        )

    model.forward_batched = fake_forward_batched
    losses = model.loss([
        SimpleNamespace(eviction_steps=[high_wrong]),
        SimpleNamespace(eviction_steps=[low_wrong]),
        SimpleNamespace(eviction_steps=[top_correct]),
    ])
    expected = torch.nn.functional.softplus(
        torch.tensor(1.5) - torch.tensor(0.0) + 0.2
    )

    assert torch.allclose(losses["lcp_wrong_margin"], expected)
    assert model.last_loss_stats["lcp_wrong_cases_count"] == 2
    assert model.last_loss_stats["lcp_wrong_high_lcp_count"] == 1
    assert model.last_loss_stats["lcp_wrong_margin_count"] == 1


def test_loss_accepts_tuple_lcp_diagnostics_for_stats():
    model = make_model(reuse_loss_weight=0.0)
    leaf_paths = [(1, 2, 3), (4,), (1, 2, 8)]
    step = make_step(
        leaf_paths,
        [float("inf"), 1.0, 2.0],
        lcp_diagnostics=TrieParrotModel.lcp_features_from_paths(
            leaf_paths,
            (1, 2, 9),
        ),
    )

    losses = model.loss([SimpleNamespace(eviction_steps=[step])])

    assert torch.isfinite(sum(losses.values()))
    assert model.last_loss_stats["oracle_target_lcp_lcp_len_mean"] == 2.0
    assert model.last_loss_stats["oracle_target_lcp_count"] == 1


def test_eviction_decision_steps_train_only_when_enabled():
    step = make_step(
        [(1,), (2,), (3,)],
        [1.0, 2.0, float("inf")],
        step_kind="eviction_decision",
    )
    snapshot = SimpleNamespace(eviction_steps=[step])

    default_model = make_model()
    default_losses = default_model.loss([snapshot])
    assert default_model.last_loss_stats["loss_steps"] == 0
    assert default_model.last_loss_stats["eviction_decision_steps"] == 0
    assert all(value.item() == 0.0 for value in default_losses.values())

    enabled_model = make_model(
        train_on_eviction_decision=True,
        eviction_decision_loss_weight=4.0,
    )
    enabled_losses = enabled_model.loss([snapshot])
    assert enabled_model.last_loss_stats["loss_steps"] == 1
    assert enabled_model.last_loss_stats["eviction_decision_steps"] == 1
    assert enabled_model.last_loss_stats["microstep_access_steps"] == 0
    assert torch.isfinite(sum(enabled_losses.values()))


def test_invalid_ce_target_policy_rejected():
    try:
        make_model(ce_target_policy="bogus")
    except ValueError as exc:
        assert "ce_target_policy" in str(exc)
    else:
        raise AssertionError("invalid ce_target_policy should fail fast")


def test_required_snapshot_fields_are_enforced():
    model = make_model()
    base = dict(
        leaf_paths=[(1,), (2,)],
        oracle_distances=[1.0, float("inf")],
        oracle_target=1,
        num_candidates=2,
    )
    for missing_field in (
        "microstep_history_paths",
        "request_history_paths",
        "lru_features",
    ):
        step_kwargs = dict(base)
        if missing_field != "microstep_history_paths":
            step_kwargs["microstep_history_paths"] = ((9,),)
        if missing_field != "request_history_paths":
            step_kwargs["request_history_paths"] = ((7,),)
        if missing_field != "lru_features":
            step_kwargs["lru_features"] = lru_features_for(step_kwargs["leaf_paths"])

        try:
            model.loss([SimpleNamespace(eviction_steps=[SimpleNamespace(**step_kwargs)])])
        except ValueError as exc:
            assert missing_field in str(exc)
        else:
            raise AssertionError(f"{missing_field} should be required")


def test_lru_feature_width_is_strict():
    model = make_model()
    step = SimpleNamespace(
        leaf_paths=[(1,), (2,)],
        oracle_distances=[1.0, float("inf")],
        oracle_target=1,
        num_candidates=2,
        microstep_history_paths=((9,),),
        request_history_paths=((7,),),
        lru_features=((1.0, 1.0), (2.0, 2.0)),
    )

    try:
        model.loss([SimpleNamespace(eviction_steps=[step])])
    except ValueError as exc:
        assert "lru_features width" in str(exc)
    else:
        raise AssertionError("lru_features width should be strict")


def test_empty_history_paths_are_not_real_history_slots():
    model = make_model()
    device = torch.device("cpu")

    assert model._encode_history_paths(None, device) is None
    assert model._encode_history_paths((), device) is None

    memory, has_history = model._prepare_history_memory(None, device)
    assert memory.shape == (1, model.hidden_size)
    assert not has_history


def test_batched_path_encoding_matches_stepwise_encoding():
    model = make_model()
    device = torch.device("cpu")
    paths = [(), (1,), (1, 2), (1,), (3, 4, 5), (1, 2)]

    expected = torch.cat(
        [model._encode_path(path, device) for path in paths],
        dim=0,
    )
    actual = model._encode_path_batch(paths, device)
    deduplicated = model._encode_path_batch(paths, device, deduplicate=True)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(deduplicated, expected, atol=1e-6)


def test_deduplicated_path_encoding_backward_matches_full_batch():
    torch.manual_seed(123)
    full_model = make_model()
    dedup_model = make_model()
    dedup_model.load_state_dict(full_model.state_dict())
    device = torch.device("cpu")
    paths = [(7,), (7,), (7, 8), (9,), (7, 8), (7,)]

    full_encoded = full_model._encode_path_batch(paths, device)
    dedup_encoded = dedup_model._encode_path_batch(
        paths,
        device,
        deduplicate=True,
    )

    assert torch.allclose(dedup_encoded, full_encoded, atol=1e-6)

    full_encoded.pow(2).sum().backward()
    dedup_encoded.pow(2).sum().backward()

    for (full_name, full_param), (dedup_name, dedup_param) in zip(
        full_model.named_parameters(),
        dedup_model.named_parameters(),
    ):
        assert full_name == dedup_name
        if full_param.grad is None:
            assert dedup_param.grad is None
            continue
        assert dedup_param.grad is not None, full_name
        assert torch.isfinite(dedup_param.grad).all(), full_name
        assert torch.allclose(
            dedup_param.grad,
            full_param.grad,
            atol=1e-6,
        ), full_name


def test_forward_path_cache_reuses_shared_prefix_states_once():
    model = make_model()
    device = torch.device("cpu")
    paths = [(1, 2, 3, 4), (1, 2, 5, 6)]
    expected = model._encode_path_batch(paths, device)
    cache = model._new_path_encoding_cache(device)
    lstm_batch_rows = []

    def capture_lstm_batch(module, inputs, output):
        del module, output
        lstm_batch_rows.append(inputs[0].shape[0])

    hook = model.path_lstm.register_forward_hook(capture_lstm_batch)
    try:
        actual = model._encode_path_batch(paths, device, cache=cache)
        cached_again = model._encode_path_batch([paths[0]], device, cache=cache)
    finally:
        hook.remove()

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(cached_again, expected[:1], atol=1e-6)
    assert lstm_batch_rows == [1, 1, 2, 2]


def test_forward_batched_cache_is_shared_across_candidate_and_histories():
    torch.manual_seed(456)
    reference_model = make_model()
    cached_model = make_model()
    cached_model.load_state_dict(reference_model.state_dict())
    device = torch.device("cpu")

    shared_path = (1, 2, 3)
    sibling_path = (1, 2, 4)
    microstep_history_paths_batch = [
        (shared_path, sibling_path),
        (shared_path,),
    ]
    candidate_paths_batch = [
        [shared_path, sibling_path],
        [shared_path, sibling_path],
    ]
    request_history_paths_batch = [
        (shared_path,),
        (shared_path, sibling_path),
    ]
    lru_features_batch = [
        lru_features_for(candidate_paths)
        for candidate_paths in candidate_paths_batch
    ]

    ref_micro, ref_micro_mask = reference_model._encode_history_paths_batch(
        microstep_history_paths_batch,
        device,
        max_history=reference_model.max_microstep_history,
    )
    ref_request, ref_request_mask = reference_model._encode_request_history_paths_batch(
        request_history_paths_batch,
        device,
    )
    ref_candidates, ref_candidate_mask = reference_model._encode_candidate_paths_batch(
        candidate_paths_batch,
        device,
    )
    ref_lru = reference_model._prepare_lru_features_batch(
        lru_features_batch,
        ref_candidate_mask,
        device,
    )
    ref_logits, ref_reuse = reference_model._forward_batched_encoded(
        ref_micro,
        ref_micro_mask,
        ref_request,
        ref_request_mask,
        ref_candidates,
        ref_candidate_mask,
        ref_lru,
    )

    lstm_batch_rows = []

    def capture_lstm_batch(module, inputs, output):
        del module, output
        lstm_batch_rows.append(inputs[0].shape[0])

    hook = cached_model.path_lstm.register_forward_hook(capture_lstm_batch)
    try:
        cached_logits, cached_reuse, cached_mask = cached_model.forward_batched(
            microstep_history_paths_batch,
            candidate_paths_batch,
            request_history_paths_batch,
            lru_features_batch,
        )
    finally:
        hook.remove()

    assert cached_mask.tolist() == ref_candidate_mask.tolist()
    assert torch.allclose(cached_logits, ref_logits, atol=1e-6)
    assert torch.allclose(cached_reuse, ref_reuse, atol=1e-6)
    assert lstm_batch_rows == [1, 1, 2]

    ref_objective = (
        ref_logits.masked_select(ref_candidate_mask).sum()
        + ref_reuse.masked_select(ref_candidate_mask).sum()
    )
    cached_objective = (
        cached_logits.masked_select(cached_mask).sum()
        + cached_reuse.masked_select(cached_mask).sum()
    )
    ref_objective.backward()
    cached_objective.backward()

    for ref_name, ref_param in reference_model.named_parameters():
        if not (
            ref_name.startswith("node_embedder.")
            or ref_name.startswith("path_lstm.")
        ):
            continue
        cached_param = dict(cached_model.named_parameters())[ref_name]
        assert ref_param.grad is not None, ref_name
        assert cached_param.grad is not None, ref_name
        assert torch.allclose(cached_param.grad, ref_param.grad, atol=1e-6), ref_name


def test_loss_backward_keeps_path_cache_differentiable_and_forward_local():
    model = make_model(reuse_loss_weight=0.2, ce_loss_weight=0.5)
    snapshot = SimpleNamespace(
        eviction_steps=[
            make_step(
                [(1, 2, 3), (1, 2, 4), (1, 5)],
                [1.0, 10.0, float("inf")],
                microstep_history_paths=((1, 2, 3), (1, 2, 4)),
                request_history_paths=((1, 2, 3),),
            ),
            make_step(
                [(1, 2, 3), (1, 6), (7,)],
                [float("inf"), 2.0, 1.0],
                microstep_history_paths=((1, 2, 3),),
                request_history_paths=((1, 2, 3), (1, 6)),
            ),
        ]
    )

    losses = model.loss([snapshot])
    total = sum(losses.values())
    assert total.requires_grad
    total.backward()

    assert not hasattr(model, "_active_path_encoding_cache")
    assert model.node_embedder.embedding.weight.grad is not None
    assert model.path_lstm.gates.weight.grad is not None
    assert torch.isfinite(model.node_embedder.embedding.weight.grad).all()
    assert torch.isfinite(model.path_lstm.gates.weight.grad).all()
    assert model.node_embedder.embedding.weight.grad.abs().sum() > 0
    assert model.path_lstm.gates.weight.grad.abs().sum() > 0


def test_history_batch_deduplicates_before_path_lstm_encoding():
    model = make_model()
    device = torch.device("cpu")
    embedded_shapes = []

    def capture_embedding_input(module, inputs, output):
        del module, output
        embedded_shapes.append(tuple(inputs[0].shape))

    hook = model.node_embedder.register_forward_hook(capture_embedding_input)
    try:
        memory, mask = model._encode_request_history_paths_batch(
            [
                ((7,),),
                ((7,),),
                ((8,), (7,)),
            ],
            device,
        )
    finally:
        hook.remove()

    assert memory.shape == (3, 2, model.hidden_size)
    assert mask.tolist() == [[True, False], [True, False], [True, True]]
    assert embedded_shapes == [(2, 1)]


def test_candidate_batch_keeps_default_non_deduplicated_encoding():
    model = make_model()
    device = torch.device("cpu")
    embedded_shapes = []

    def capture_embedding_input(module, inputs, output):
        del module, output
        embedded_shapes.append(tuple(inputs[0].shape))

    hook = model.node_embedder.register_forward_hook(capture_embedding_input)
    try:
        candidates, mask = model._encode_candidate_paths_batch(
            [
                [(7,), (7,)],
                [(8,), (7,)],
            ],
            device,
        )
    finally:
        hook.remove()

    assert candidates.shape == (2, 2, model.hidden_size)
    assert mask.tolist() == [[True, True], [True, True]]
    assert embedded_shapes == [(4, 1)]


def test_from_config_reads_lru_trie_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "vocab_size": 128,
                    "node_embed_dim": 16,
                    "hidden_size": 32,
                    "max_request_history": 11,
                    "max_microstep_history": 7,
                    "lru_feature_dim": 5,
                    "ce_target_policy": "top_set",
                },
                f,
            )

        model = TrieParrotModel.from_config(config_path)
        assert model.max_request_history == 11
        assert model.max_microstep_history == 7
        assert model.lru_feature_dim == 5
        assert model.ce_target_policy == "top_set"


if __name__ == "__main__":
    test_relevance_transform_parrot_style()
    test_ndcg_position_sign_improving_high_relevance_score_lowers_loss()
    test_loss_uses_all_candidates_by_default()
    test_loss_candidate_cap_keeps_current_hit_required_candidate()
    test_loss_candidate_cap_forces_lru_and_oracle_top_set_candidates()
    test_loss_warmup_matches_parrot_suffix_loss()
    test_batched_loss_matches_stepwise_reference_with_padding()
    test_loss_sum_reduction_matches_mean_times_counts()
    test_from_config_lru_prior_alpha_fields()
    test_old_lru_head_checkpoint_migrates_to_lru_prior()
    test_loss_batches_same_time_steps_across_windows()
    test_loss_has_finite_reuse_with_inf_distance()
    test_ce_optional_default_zero_and_enabled_path()
    test_argmax_ce_matches_single_target_cross_entropy()
    test_top_set_ce_targets_all_max_relevance_candidates()
    test_top_set_ce_loss_uses_logsumexp_top_set_formula()
    test_hard_lru_margin_only_active_when_lru_not_in_top_set()
    test_lcp_wrong_margin_only_active_for_high_lcp_wrong_target()
    test_loss_accepts_tuple_lcp_diagnostics_for_stats()
    test_eviction_decision_steps_train_only_when_enabled()
    test_invalid_ce_target_policy_rejected()
    test_required_snapshot_fields_are_enforced()
    test_lru_feature_width_is_strict()
    test_empty_history_paths_are_not_real_history_slots()
    test_batched_path_encoding_matches_stepwise_encoding()
    test_deduplicated_path_encoding_backward_matches_full_batch()
    test_forward_path_cache_reuses_shared_prefix_states_once()
    test_forward_batched_cache_is_shared_across_candidate_and_histories()
    test_loss_backward_keeps_path_cache_differentiable_and_forward_local()
    test_history_batch_deduplicates_before_path_lstm_encoding()
    test_candidate_batch_keeps_default_non_deduplicated_encoding()
    test_from_config_reads_lru_trie_fields()
    print("TRIE-PARROT NDCG LOSS TESTS PASSED")
