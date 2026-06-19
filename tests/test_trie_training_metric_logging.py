#!/usr/bin/env python3
"""Tests for TrieParrot training metric logging and automatic loss plotting."""
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trie_model.__main__ import (
    METRIC_FIELDS,
    append_metric_row,
    as_microstep_window_batches,
    count_microstep_steps,
    count_microstep_windows,
    compute_rank_eval_metrics,
    plot_loss_curves,
    round_collection_examples,
    round_step_budget,
    should_run_periodic_event,
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


if __name__ == "__main__":
    test_metric_append_writes_header_and_preserves_existing_rows()
    test_plot_loss_curves_filters_run_id_and_writes_png()
    test_plot_loss_curves_handles_single_step_run()
    test_round_budget_limits_each_collect_round()
    test_rank_eval_metrics_are_logged_fields_and_finite()
    test_microstep_window_batches_are_32_by_40_and_consecutive()
    test_full_dagger_config_uses_lru_trie_fields()
    test_full_parrot_like_config_uses_parrot_window_shape()
    print("TRIE TRAINING METRIC LOGGING TESTS PASSED")
