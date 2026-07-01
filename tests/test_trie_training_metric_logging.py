#!/usr/bin/env python3
"""Tests for TrieParrot training metric logging and automatic loss plotting."""
import csv
import json
import math
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from model.trie_model.__main__ import (
    METRIC_FIELDS,
    append_metric_row,
    as_microstep_window_batches,
    count_microstep_steps,
    count_microstep_windows,
    combine_loss_stats,
    create_frozen_cpu_collection_model,
    create_trie_parrot_model_from_config,
    compute_rank_eval_metrics,
    compute_training_losses,
    freeze_model_state_dict_for_collection,
    iter_loss_microbatches,
    plot_loss_curves,
    plan_collection_round,
    round_collection_examples,
    round_step_budget,
    resolve_bool_config,
    resolve_training_round_budget,
    scheduled_loss_weight,
    should_run_periodic_event,
    latest_training_checkpoint,
    parse_train_device_ids,
    summarize_loss_batch,
    submit_async_collection,
    split_batch_for_devices,
    tune_collection_multiplier,
    training_checkpoint_step,
    wait_for_async_collection,
)
from model.trie_model.model import TrieParrotModel
from types import SimpleNamespace


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_metric_append_writes_header_and_preserves_existing_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "training_metrics.csv")

        append_metric_row(path, {
            "run_id": "run_a",
            "event": "train_step",
            "step": 0,
            "loss_total": 1.0,
        })
        append_metric_row(path, {
            "run_id": "run_b",
            "event": "eval",
            "step": 1,
            "eval_hr": 0.25,
            "training_checkpoint_path": "training_step_1.pt",
        })

        with open(path, newline="") as f:
            header = f.readline().strip().split(",")
        rows = read_rows(path)

        assert header == METRIC_FIELDS
        assert len(rows) == 2
        assert rows[0]["run_id"] == "run_a"
        assert rows[0]["event"] == "train_step"
        assert rows[0]["loss_total"] == "1.0"
        assert rows[1]["run_id"] == "run_b"
        assert rows[1]["event"] == "eval"
        assert rows[1]["eval_hr"] == "0.25"
        assert rows[1]["training_checkpoint_path"] == "training_step_1.pt"


def test_metric_append_upgrades_older_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "training_metrics.csv")
        newly_added_fields = {
            "collection_seconds",
            "collection_wait_seconds",
            "train_round_seconds",
            "collection_train_time_ratio",
            "async_collection",
            "collection_multiplier",
            "collection_autotune",
            "collection_target_train_time_ratio",
            "max_collection_requests",
            "max_collection_snapshots",
            "optimizer_steps_per_collection",
        }
        old_fields = [
            field for field in METRIC_FIELDS
            if field not in (
                {"num_microsteps", "training_checkpoint_path"}
                | newly_added_fields
            )
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({
                "run_id": "old_run",
                "event": "train_step",
                "step": 7,
                "num_snapshots": 123,
            })

        append_metric_row(path, {
            "run_id": "new_run",
            "event": "checkpoint",
            "step": 8,
            "num_microsteps": 456,
            "training_checkpoint_path": "training_step_8.pt",
        })

        with open(path, newline="") as f:
            header = f.readline().strip().split(",")
        rows = read_rows(path)

        assert header == METRIC_FIELDS
        assert len(rows) == 2
        assert rows[0]["run_id"] == "old_run"
        assert rows[0]["num_snapshots"] == "123"
        assert rows[0]["num_microsteps"] == ""
        assert rows[1]["run_id"] == "new_run"
        assert rows[1]["num_microsteps"] == "456"
        assert rows[1]["training_checkpoint_path"] == "training_step_8.pt"


def test_latest_training_checkpoint_includes_final_training_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        step_100 = os.path.join(tmpdir, "training_step_100.pt")
        step_200 = os.path.join(tmpdir, "training_step_200.pt")
        final_300 = os.path.join(tmpdir, "training_final_300.pt")
        model_only = os.path.join(tmpdir, "final_999.ckpt")

        for path in (step_100, step_200, final_300, model_only):
            with open(path, "wb"):
                pass

        assert training_checkpoint_step(step_100) == 100
        assert training_checkpoint_step(final_300) == 300
        assert latest_training_checkpoint(tmpdir) == final_300


def test_train_device_parser_and_batch_splitter():
    assert parse_train_device_ids(None, torch.device("cuda:4"), 8) == [4]
    assert parse_train_device_ids("gpu4,cuda:5,4", torch.device("cuda:4"), 8) == [4, 5]
    assert parse_train_device_ids("auto", torch.device("cuda:4"), 6) == [4, 0, 1, 2, 3, 5]
    assert split_batch_for_devices(list(range(10)), 4) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
        [8, 9],
    ]

    try:
        parse_train_device_ids("0", torch.device("cpu"), 8)
    except ValueError as exc:
        assert "--train_devices requires" in str(exc)
    else:
        raise AssertionError("CPU training must reject explicit train_devices")


def test_plot_loss_curves_filters_run_id_and_writes_png():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, "training_metrics.csv")
        output_path = os.path.join(tmpdir, "loss_curves_run_b.png")

        append_metric_row(metrics_path, {
            "run_id": "run_a",
            "event": "train_step",
            "step": 0,
            "loss_total": 99.0,
            "loss_ranking": 98.0,
            "loss_reuse": 1.0,
            "loss_ce": 0.0,
        })
        for step in range(3):
            append_metric_row(metrics_path, {
                "run_id": "run_b",
                "event": "train_step",
                "step": step,
                "loss_total": 3.0 - step,
                "loss_ranking": 2.5 - step * 0.5,
                "loss_reuse": 0.2,
                "loss_ce": 0.0,
            })
        append_metric_row(metrics_path, {
            "run_id": "run_b",
            "event": "eval",
            "step": 1,
            "eval_hr": 0.3,
        })

        ok = plot_loss_curves(metrics_path, "run_b", output_path)

        assert ok
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_plot_loss_curves_handles_single_step_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, "training_metrics.csv")
        output_path = os.path.join(tmpdir, "loss_curves_single.png")

        append_metric_row(metrics_path, {
            "run_id": "single",
            "event": "train_step",
            "step": 0,
            "loss_total": 1.0,
            "loss_ranking": -0.5,
            "loss_reuse": 1.5,
            "loss_ce": 0.0,
        })

        ok = plot_loss_curves(metrics_path, "single", output_path)

        assert ok
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_round_budget_limits_each_collect_round():
    assert round_step_budget(step=0, total_steps=300, dagger_update_freq=50) == 50
    assert round_step_budget(step=260, total_steps=300, dagger_update_freq=50) == 40
    assert round_step_budget(step=300, total_steps=300, dagger_update_freq=50) == 0
    assert round_collection_examples(
        step=0,
        total_steps=300,
        dagger_update_freq=50,
        batch_size=2,
        collection_multiplier=4,
    ) == 400
    assert round_collection_examples(
        step=260,
        total_steps=300,
        dagger_update_freq=50,
        batch_size=2,
        collection_multiplier=4,
    ) == 320
    assert round_collection_examples(
        step=300,
        total_steps=300,
        dagger_update_freq=50,
        batch_size=2,
        collection_multiplier=4,
    ) == 0
    assert round_collection_examples(
        step=0,
        total_steps=300,
        dagger_update_freq=50,
        batch_size=2,
        collection_multiplier=0,
    ) == 100
    assert should_run_periodic_event(step=50, freq=50)
    assert not should_run_periodic_event(step=0, freq=50)
    assert not should_run_periodic_event(step=50, freq=0)


def test_async_collection_config_and_round_planning():
    assert resolve_bool_config({"async_collection": "true"}, "async_collection")
    assert not resolve_bool_config({"async_collection": "false"}, "async_collection", True)

    plan = plan_collection_round(
        step=10,
        total_steps=100,
        dagger_update_freq=20,
        batch_size=4,
        collection_multiplier=3,
        collection_snapshot_cap="round_budget",
    )

    assert plan.step == 10
    assert plan.collection_multiplier == 3
    assert plan.round_budget == 20
    assert plan.max_examples == 240
    assert not plan.consume_all_collected_snapshots

    snapshots = [SimpleNamespace(eviction_steps=[object() for _ in range(12)])]
    assert resolve_training_round_budget(
        plan,
        snapshots,
        batch_size=4,
        sequence_length=3,
        total_steps=100,
        optimizer_steps_per_collection=5,
    ) == 5

    capped_plan = plan_collection_round(
        step=10,
        total_steps=100,
        dagger_update_freq=20,
        batch_size=4,
        collection_multiplier=3,
        collection_snapshot_cap=12,
    )
    assert capped_plan.max_examples == 12
    assert capped_plan.consume_all_collected_snapshots
    assert resolve_training_round_budget(
        capped_plan,
        snapshots,
        batch_size=4,
        sequence_length=3,
        total_steps=100,
        optimizer_steps_per_collection=None,
    ) == 3


def test_collection_multiplier_autotune_moves_toward_target_ratio():
    assert tune_collection_multiplier(
        current_multiplier=4,
        collection_seconds=1.0,
        train_round_seconds=4.0,
        target_ratio=1.0,
        min_multiplier=1,
        max_multiplier=16,
        max_scale=2.0,
    ) == 8
    assert tune_collection_multiplier(
        current_multiplier=8,
        collection_seconds=8.0,
        train_round_seconds=2.0,
        target_ratio=1.0,
        min_multiplier=1,
        max_multiplier=16,
        max_scale=2.0,
    ) == 4
    assert tune_collection_multiplier(
        current_multiplier=4,
        collection_seconds=0.0,
        train_round_seconds=2.0,
        target_ratio=1.0,
        min_multiplier=1,
        max_multiplier=16,
    ) == 4
    assert tune_collection_multiplier(
        current_multiplier=4,
        collection_seconds=1.0,
        train_round_seconds=2.0,
        target_ratio=None,
        min_multiplier=1,
        max_multiplier=16,
    ) == 4


def test_frozen_cpu_collection_model_is_isolated_from_training_model():
    config = {
        "vocab_size": 32,
        "node_embed_dim": 8,
        "hidden_size": 12,
        "max_attention_history": 4,
        "max_request_history": 4,
        "max_microstep_history": 4,
        "lru_feature_dim": 5,
    }
    model = create_trie_parrot_model_from_config(config, vocab_size=32)
    frozen_state = freeze_model_state_dict_for_collection(model)
    first_key = next(iter(frozen_state))
    frozen_first_param = frozen_state[first_key].clone()

    with torch.no_grad():
        next(model.parameters()).add_(10.0)

    collection_model = create_frozen_cpu_collection_model(
        config,
        vocab_size=32,
        frozen_state_dict=frozen_state,
    )

    assert not collection_model.training
    assert all(param.device.type == "cpu" for param in collection_model.parameters())
    assert all(not param.requires_grad for param in collection_model.parameters())
    assert torch.allclose(collection_model.state_dict()[first_key], frozen_first_param)
    assert not torch.allclose(model.state_dict()[first_key], frozen_first_param)


def test_async_collection_worker_uses_frozen_cpu_snapshot():
    config = {
        "vocab_size": 32,
        "node_embed_dim": 8,
        "hidden_size": 12,
        "max_attention_history": 4,
        "max_request_history": 4,
        "max_microstep_history": 4,
        "lru_feature_dim": 5,
    }
    model = create_trie_parrot_model_from_config(config, vocab_size=32)
    frozen_first_param = next(model.parameters()).detach().cpu().clone()
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def fake_collect(
        data_path,
        vocab_path,
        collection_model,
        max_node_num,
        model_prob,
        max_examples,
        max_requests,
        train_on_eviction_decision=False,
    ):
        seen["device"] = next(collection_model.parameters()).device.type
        seen["requires_grad"] = any(
            param.requires_grad for param in collection_model.parameters()
        )
        seen["first_param"] = next(collection_model.parameters()).detach().cpu().clone()
        seen["model_prob"] = model_prob
        seen["max_examples"] = max_examples
        seen["train_on_eviction_decision"] = train_on_eviction_decision
        started.set()
        release.wait(timeout=5)
        return [SimpleNamespace(eviction_steps=[SimpleNamespace()])], 0.75

    plan = plan_collection_round(
        step=0,
        total_steps=10,
        dagger_update_freq=2,
        batch_size=4,
        collection_multiplier=2,
        collection_snapshot_cap="round_budget",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = submit_async_collection(
            executor,
            plan,
            "train.pkl",
            "vocab.json",
            model,
            config,
            32,
            8,
            0.5,
            3,
            collect_fn=fake_collect,
        )
        assert started.wait(timeout=5)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        release.set()
        result = wait_for_async_collection(job)

    assert result.async_collection
    assert result.plan is plan
    assert result.train_hit_rate == 0.75
    assert result.collection_seconds >= 0.0
    assert result.collection_wait_seconds >= 0.0
    assert seen["device"] == "cpu"
    assert not seen["requires_grad"]
    assert seen["model_prob"] == 0.5
    assert seen["max_examples"] == 16
    assert not seen["train_on_eviction_decision"]
    assert torch.allclose(seen["first_param"], frozen_first_param)
    assert not torch.allclose(next(model.parameters()).detach().cpu(), frozen_first_param)


def test_rank_eval_metrics_are_logged_fields_and_finite():
    model = TrieParrotModel(vocab_size=128, node_embed_dim=16, hidden_size=32)
    step = SimpleNamespace(
        leaf_paths=[(1,), (2,), (3,)],
        microstep_history_paths=((9,), (9, 8)),
        request_history_paths=((7,),),
        lru_features=(
            (1.0, 1.0, 1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0, 2.0, 1.0),
            (3.0, 3.0, 3.0, 3.0, 1.0),
        ),
        oracle_distances=[1.0, 10.0, float("inf")],
        oracle_target=2,
        num_candidates=3,
    )

    metrics = compute_rank_eval_metrics(model, [step])

    for key in [
        "rank_eval_ndcg",
        "rank_eval_pairwise_acc",
        "rank_eval_top1_acc",
        "rank_eval_score_std",
        "rank_eval_steps",
        "rank_eval_pairs",
    ]:
        assert key in METRIC_FIELDS
        assert key in metrics

    assert metrics["rank_eval_steps"] == 1
    assert metrics["rank_eval_pairs"] > 0
    assert 0.0 <= metrics["rank_eval_ndcg"] <= 1.0
    assert 0.0 <= metrics["rank_eval_pairwise_acc"] <= 1.0
    assert 0.0 <= metrics["rank_eval_top1_acc"] <= 1.0
    assert metrics["rank_eval_score_std"] >= 0.0


def test_microstep_window_batches_are_32_by_40_and_consecutive():
    steps = [
        SimpleNamespace(step_id=idx, num_candidates=2)
        for idx in range(80)
    ]
    snapshots = [
        SimpleNamespace(eviction_steps=steps[:25]),
        SimpleNamespace(eviction_steps=steps[25:]),
    ]

    batches = list(as_microstep_window_batches(
        snapshots,
        batch_size=32,
        sequence_length=40,
        shuffle=False,
    ))

    assert count_microstep_steps(snapshots) == 80
    assert count_microstep_windows(snapshots, 40) == 41
    assert len(batches) == 2
    assert len(batches[0]) == 32
    assert len(batches[1]) == 9
    first_window = batches[0][0].eviction_steps
    last_window = batches[-1][-1].eviction_steps
    assert [step.step_id for step in first_window] == list(range(40))
    assert [step.step_id for step in last_window] == list(range(40, 80))


def make_loss_step(leaf_paths, oracle_distances, required_candidate_indices=None):
    lru_features = []
    for idx, path in enumerate(leaf_paths):
        age = float(idx + 1)
        lru_features.append((age, age, age, age, float(len(path))))
    return SimpleNamespace(
        leaf_paths=list(leaf_paths),
        oracle_distances=list(oracle_distances),
        oracle_target=max(
            range(len(oracle_distances)),
            key=lambda idx: oracle_distances[idx],
        ),
        required_candidate_indices=required_candidate_indices,
        microstep_history_paths=((9,), (9, 8)),
        request_history_paths=((7,),),
        lru_features=tuple(lru_features),
        num_candidates=len(leaf_paths),
    )


def test_loss_microbatch_sum_matches_full_batch_mean():
    model = TrieParrotModel(
        vocab_size=128,
        node_embed_dim=16,
        hidden_size=32,
        reuse_loss_weight=1.0,
    )
    batch = [
        SimpleNamespace(eviction_steps=[
            make_loss_step([(1,), (2,), (3,), (4,), (5,)], [0.0, 2.0, 3.0, 4.0, float("inf")], (0,)),
            make_loss_step([(6,), (7,), (8,)], [1.0, float("inf"), 3.0]),
            make_loss_step([(9,), (10,)], [float("inf"), 2.0]),
        ]),
        SimpleNamespace(eviction_steps=[
            make_loss_step([(11,), (12,), (13,), (14,)], [1.0, 2.0, float("inf"), 4.0]),
            make_loss_step([(15,), (16,), (17,)], [float("inf"), 1.0, 2.0]),
            make_loss_step([(18,), (19,), (20,)], [2.0, 3.0, float("inf")]),
        ]),
        SimpleNamespace(eviction_steps=[
            make_loss_step([(21,), (22,), (23,)], [2.0, float("inf"), 1.0]),
            make_loss_step([(24,), (25,), (26,), (27,)], [1.0, 2.0, 3.0, float("inf")]),
            make_loss_step([(28,), (29,)], [1.0, float("inf")]),
        ]),
        SimpleNamespace(eviction_steps=[
            make_loss_step([(30,), (31,), (32,), (33,), (34,)], [5.0, 4.0, 3.0, 2.0, float("inf")], (0,)),
            make_loss_step([(35,), (36,), (37,)], [1.0, 2.0, float("inf")]),
            make_loss_step([(38,), (39,), (40,)], [float("inf"), 2.0, 1.0]),
        ]),
    ]

    full_losses = compute_training_losses(
        model,
        batch,
        max_candidates=3,
        max_steps_per_snapshot=None,
        warmup_steps_per_snapshot=1,
        train_device_ids=[],
    )
    full_stats = dict(model.last_loss_stats)
    total_stats = summarize_loss_batch(
        model,
        batch,
        max_candidates=3,
        max_steps_per_snapshot=None,
        warmup_steps_per_snapshot=1,
    )
    for key in [
        "full_steps",
        "capped_steps",
        "candidate_count",
        "ranking_count",
        "reuse_count",
        "ce_count",
        "top_set_ce_count",
        "hard_lru_margin_count",
        "lcp_wrong_margin_count",
        "warmup_steps",
        "loss_steps",
        "microstep_access_steps",
        "eviction_decision_steps",
        "lru_target_kept_count",
        "oracle_top_set_kept_count",
        "hard_lru_cases_count",
        "max_loss_candidates_effective",
    ]:
        assert total_stats[key] == full_stats[key], key

    loss_names = model.loss_names()
    micro_losses = {name: torch.tensor(0.0) for name in loss_names}
    micro_stats = []
    for micro_batch in iter_loss_microbatches(batch, 2):
        loss_sums = compute_training_losses(
            model,
            micro_batch,
            max_candidates=3,
            max_steps_per_snapshot=None,
            warmup_steps_per_snapshot=1,
            train_device_ids=[],
            reduction="sum",
        )
        micro_stats.append(dict(model.last_loss_stats))
        for name in loss_names:
            count = total_stats[f"{name}_count"]
            if count > 0:
                micro_losses[name] = micro_losses[name] + loss_sums[name] / count

    for name in loss_names:
        assert torch.allclose(micro_losses[name], full_losses[name], atol=1e-6)

    combined_stats = combine_loss_stats(micro_stats)
    for key in [
        "top_set_acc",
        "regret",
    ]:
        assert math.isclose(combined_stats[key], full_stats[key], abs_tol=1e-8), key
    for key in [
        "lru_target_kept_count",
        "oracle_top_set_kept_count",
        "max_loss_candidates_effective",
    ]:
        assert combined_stats[key] == full_stats[key], key


def test_full_dagger_config_uses_lru_trie_fields():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "full_dagger_oasst1_b16_c256.json",
    )
    with open(config_path) as f:
        config = json.load(f)

    assert "candidate_scorer_mode" not in config
    assert config.get("max_request_history") == 30
    assert config.get("max_microstep_history") == 30
    assert config.get("lru_feature_dim") == 5


def test_full_parrot_like_config_uses_parrot_window_shape():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "full_parrot_like_oasst1_b16_c256.json",
    )
    with open(config_path) as f:
        config = json.load(f)

    assert config.get("batch_size") == 16
    assert config.get("sequence_length") == 40
    assert "candidate_scorer_mode" not in config
    assert config.get("max_request_history") == 30
    assert config.get("max_microstep_history") == 30
    assert config.get("lru_feature_dim") == 5


def test_phase_configs_capture_hard_lru_ablation_knobs():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
    )
    with open(os.path.join(config_dir, "full_parrot_like_phase1_oasst1_b16_c256.json")) as f:
        phase1 = json.load(f)
    with open(os.path.join(config_dir, "full_parrot_like_phase2_oasst1_b16_c256.json")) as f:
        phase2 = json.load(f)

    for config in (phase1, phase2):
        assert config["max_loss_candidates"] == 32
        assert config["lru_prior_alpha_init"] == 0.75
        assert config["lru_prior_alpha_max"] == 1.5
        assert config["lru_prior_alpha_learnable"] is True

    assert phase1["train_on_eviction_decision"] is False
    assert phase1["top_set_ce_weight"] == 0.0
    assert phase1["hard_lru_margin_weight"] == 0.0

    assert phase2["train_on_eviction_decision"] is True
    assert phase2["eviction_decision_loss_weight"] == 4.0
    assert phase2["microstep_access_loss_weight"] == 0.25
    assert phase2["top_set_ce_weight"] == 1.0
    assert phase2["hard_lru_margin_weight"] == 1.0
    assert phase2["hard_lru_margin"] == 0.2


def test_phase2_lcp_config_enables_lcp_knobs_and_metrics_fields():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "full_parrot_like_phase2_lcp_oasst1_b16_c256.json",
    )
    with open(config_path) as f:
        config = json.load(f)

    assert config["eval_freq"] == 1000
    assert config["save_freq"] == 1000
    assert config["rank_eval_freq"] == 1000
    assert config["use_lcp_features"] is True
    assert config["lru_prior_alpha_min"] == 0.25
    assert config["lcp_wrong_margin_weight"] == 0.15
    assert config["lcp_wrong_margin"] == 0.2
    assert config["lcp_wrong_ratio_threshold"] == 0.5
    assert config["loss_schedule_after"] == 5000
    assert config["loss_schedule_steps"] == 5000
    assert config["top_set_ce_weight_final"] == 0.35
    assert config["hard_lru_margin_weight_final"] == 0.35
    assert config["lcp_wrong_margin_weight_final"] == 0.05
    assert "loss_lcp_wrong_margin" in METRIC_FIELDS
    assert "lcp_wrong_margin_count" in METRIC_FIELDS
    assert "top_set_ce_weight_active" in METRIC_FIELDS
    assert "hard_lru_margin_weight_active" in METRIC_FIELDS
    assert "lcp_wrong_margin_weight_active" in METRIC_FIELDS

    model = create_trie_parrot_model_from_config(config, vocab_size=32)
    assert model.use_lcp_features
    assert model.lru_prior_alpha_min == 0.25
    assert model.lcp_wrong_margin_weight == 0.15
    assert model.lcp_wrong_margin == 0.2
    assert model.lcp_wrong_ratio_threshold == 0.5


def test_scheduled_loss_weight_linear_decay():
    assert scheduled_loss_weight(1.0, 0.35, 5000, 5000, 5000) == 1.0
    assert math.isclose(
        scheduled_loss_weight(1.0, 0.35, 7500, 5000, 5000),
        0.675,
        abs_tol=1e-12,
    )
    assert scheduled_loss_weight(1.0, 0.35, 10000, 5000, 5000) == 0.35
    assert scheduled_loss_weight(1.0, None, 10000, 5000, 5000) == 1.0


if __name__ == "__main__":
    test_metric_append_writes_header_and_preserves_existing_rows()
    test_metric_append_upgrades_older_header()
    test_latest_training_checkpoint_includes_final_training_state()
    test_train_device_parser_and_batch_splitter()
    test_plot_loss_curves_filters_run_id_and_writes_png()
    test_plot_loss_curves_handles_single_step_run()
    test_round_budget_limits_each_collect_round()
    test_async_collection_config_and_round_planning()
    test_collection_multiplier_autotune_moves_toward_target_ratio()
    test_frozen_cpu_collection_model_is_isolated_from_training_model()
    test_async_collection_worker_uses_frozen_cpu_snapshot()
    test_rank_eval_metrics_are_logged_fields_and_finite()
    test_microstep_window_batches_are_32_by_40_and_consecutive()
    test_loss_microbatch_sum_matches_full_batch_mean()
    test_full_dagger_config_uses_lru_trie_fields()
    test_full_parrot_like_config_uses_parrot_window_shape()
    test_phase_configs_capture_hard_lru_ablation_knobs()
    test_phase2_lcp_config_enables_lcp_knobs_and_metrics_fields()
    test_scheduled_loss_weight_linear_decay()
    print("TRIE TRAINING METRIC LOGGING TESTS PASSED")
