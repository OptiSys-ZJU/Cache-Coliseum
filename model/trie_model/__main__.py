"""
Training script for TrieParrotModel using DAgger (Dataset Aggregation).

Usage:
    python -m model.trie_model --dataset oasst1_timed_global_b16 --device cpu
    python -m model.trie_model --dataset oasst1_timed_global_b16 --device cuda:0
"""
import os
import json
import argparse
import glob
import csv
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import torch
import tqdm
from torch.nn.parallel import parallel_apply, replicate

from model.trie_model.model import TrieParrotModel
from cache.trie.trie_cache import TrieTrainingCache, SequenceTrieCache
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieLRUAlgorithm
from data_trace.trie_data_trace import SequenceTrieDataTrace


def get_model_prob(step: int, dagger_init: float, dagger_final: float, dagger_steps: int) -> float:
    """DAgger schedule: linear interpolation from init to final over dagger_steps."""
    fraction = min(float(step) / max(dagger_steps, 1), 1.0)
    return dagger_init + fraction * (dagger_final - dagger_init)


def collect_snapshots(
    data_path: str, 
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    model_prob: float,
    max_examples: int = None,
    max_requests: int = None,
    train_on_eviction_decision: bool = False,
):
    """
    Run one pass over data, collecting microstep DAgger snapshots.
    
    Returns:
        (snapshots, hit_rate)
    """
    cache = TrieTrainingCache(
        max_node_num=max_node_num,
        model=model,
        train_on_eviction_decision=train_on_eviction_decision,
    )
    
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        # Load all sequences for oracle
        all_seqs = list(trace.iter_sequences())
    
    cache.load_future_accesses(all_seqs)
    cache.set_model_prob(model_prob)
    
    for request_idx, seq in enumerate(all_seqs):
        if max_requests is not None and request_idx >= max_requests:
            break
        cache.collect(seq)
        if (
            max_examples is not None
            and cache.collected_training_steps >= max_examples
        ):
            break
    
    return cache.get_snapshots(), cache.hit_rate


def resolve_bool_config(config: dict, key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def resolve_lru_prior_config(config: dict):
    fixed_lru_prior_alpha = config.get('lru_prior_alpha_fixed')
    fixed_lru_prior_requested = (
        fixed_lru_prior_alpha
        if isinstance(fixed_lru_prior_alpha, bool)
        else fixed_lru_prior_alpha is not None
    )
    default_lru_prior_alpha = (
        0.75
        if isinstance(fixed_lru_prior_alpha, bool)
        or fixed_lru_prior_alpha is None
        else fixed_lru_prior_alpha
    )
    return (
        config.get('lru_prior_alpha_init', default_lru_prior_alpha),
        config.get('lru_prior_alpha_max', 1.5),
        config.get('lru_prior_alpha_learnable', not fixed_lru_prior_requested),
    )


def create_trie_parrot_model_from_config(config: dict, vocab_size: int):
    (
        lru_prior_alpha_init,
        lru_prior_alpha_max,
        lru_prior_alpha_learnable,
    ) = resolve_lru_prior_config(config)
    return TrieParrotModel(
        vocab_size=vocab_size,
        node_embed_dim=config.get('node_embed_dim', 64),
        hidden_size=config.get('hidden_size', 128),
        max_attention_history=config.get('max_attention_history', 30),
        max_request_history=config.get('max_request_history'),
        max_microstep_history=config.get('max_microstep_history'),
        lru_feature_dim=config.get('lru_feature_dim', 5),
        ranking_loss_weight=config.get('ranking_loss_weight', 1.0),
        reuse_loss_weight=config.get('reuse_loss_weight', 0.1),
        ce_loss_weight=config.get('ce_loss_weight', 0.0),
        ce_target_policy=config.get('ce_target_policy', 'argmax'),
        top_set_ce_weight=config.get('top_set_ce_weight', 0.0),
        hard_lru_margin_weight=config.get('hard_lru_margin_weight', 0.0),
        hard_lru_margin=config.get('hard_lru_margin', 0.2),
        train_on_eviction_decision=resolve_bool_config(
            config,
            'train_on_eviction_decision',
            False,
        ),
        eviction_decision_loss_weight=config.get(
            'eviction_decision_loss_weight',
            1.0,
        ),
        microstep_access_loss_weight=config.get(
            'microstep_access_loss_weight',
            1.0,
        ),
        reuse_distance_log_cap=config.get('reuse_distance_log_cap', 5.0),
        ndcg_alpha=config.get('ndcg_alpha', 10.0),
        lru_prior_alpha_init=lru_prior_alpha_init,
        lru_prior_alpha_max=lru_prior_alpha_max,
        lru_prior_alpha_learnable=lru_prior_alpha_learnable,
    )


def freeze_model_state_dict_for_collection(model: TrieParrotModel):
    """Detach a training model snapshot for worker-owned CPU collection."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def create_frozen_cpu_collection_model(
    config: dict,
    vocab_size: int,
    frozen_state_dict,
):
    collection_model = create_trie_parrot_model_from_config(config, vocab_size)
    collection_model.load_state_dict_compatible(frozen_state_dict)
    collection_model.to(torch.device("cpu"))
    collection_model.eval()
    for parameter in collection_model.parameters():
        parameter.requires_grad_(False)
    return collection_model


def plan_collection_round(
    step: int,
    total_steps: int,
    dagger_update_freq: int,
    batch_size: int,
    collection_multiplier: int,
    collection_snapshot_cap,
):
    safe_collection_multiplier = max(1, int(collection_multiplier or 1))
    round_budget = round_step_budget(step, total_steps, dagger_update_freq)
    consume_all_collected_snapshots = collection_snapshot_cap != 'round_budget'
    if collection_snapshot_cap == 'round_budget':
        max_examples = round_collection_examples(
            step,
            total_steps,
            dagger_update_freq,
            batch_size,
            safe_collection_multiplier,
        )
    elif collection_snapshot_cap is None:
        max_examples = None
    else:
        max_examples = int(collection_snapshot_cap)

    return SimpleNamespace(
        step=step,
        collection_multiplier=safe_collection_multiplier,
        round_budget=round_budget,
        max_examples=max_examples,
        consume_all_collected_snapshots=consume_all_collected_snapshots,
    )


def resolve_training_round_budget(
    plan,
    snapshots,
    batch_size: int,
    sequence_length: int,
    total_steps: int,
    optimizer_steps_per_collection,
):
    round_budget = plan.round_budget
    if plan.consume_all_collected_snapshots:
        window_count = count_microstep_windows(snapshots, sequence_length)
        round_budget = min(
            math.ceil(window_count / batch_size),
            total_steps - plan.step,
        )

    if optimizer_steps_per_collection is not None:
        round_budget = min(
            round_budget,
            max(1, int(optimizer_steps_per_collection)),
            total_steps - plan.step,
        )

    return round_budget


def tune_collection_multiplier(
    current_multiplier: int,
    collection_seconds: float,
    train_round_seconds: float,
    target_ratio,
    min_multiplier: int = 1,
    max_multiplier=None,
    max_scale: float = 2.0,
) -> int:
    """
    Adjust collection volume toward a target collection/train time ratio.

    The multiplier controls collected microstep snapshots when
    max_collection_snapshots='round_budget'. Keeping collection time close to
    train time helps the async worker finish just as the GPU round ends.
    """
    current_multiplier = max(1, int(current_multiplier or 1))
    min_multiplier = max(1, int(min_multiplier or 1))
    if max_multiplier in (None, "", 0, "0", False):
        max_multiplier = current_multiplier
    max_multiplier = max(min_multiplier, int(max_multiplier))

    if target_ratio in (None, "", 0, "0", False):
        return min(max(current_multiplier, min_multiplier), max_multiplier)

    target_ratio = float(target_ratio)
    collection_seconds = float(collection_seconds or 0.0)
    train_round_seconds = float(train_round_seconds or 0.0)
    max_scale = max(1.0, float(max_scale or 1.0))
    if (
        target_ratio <= 0
        or collection_seconds <= 0
        or train_round_seconds <= 0
        or not math.isfinite(collection_seconds)
        or not math.isfinite(train_round_seconds)
    ):
        return min(max(current_multiplier, min_multiplier), max_multiplier)

    observed_ratio = collection_seconds / train_round_seconds
    if observed_ratio <= 0 or not math.isfinite(observed_ratio):
        return min(max(current_multiplier, min_multiplier), max_multiplier)

    scale = target_ratio / observed_ratio
    scale = min(max(scale, 1.0 / max_scale), max_scale)
    tuned = int(round(current_multiplier * scale))
    return min(max(max(1, tuned), min_multiplier), max_multiplier)


def collect_snapshots_with_frozen_state(
    config: dict,
    vocab_size: int,
    frozen_state_dict,
    data_path: str,
    vocab_path: str,
    max_node_num: int,
    model_prob: float,
    max_examples,
    max_requests,
    train_on_eviction_decision: bool = False,
    collect_fn=collect_snapshots,
):
    started_at = time.perf_counter()
    collection_model = create_frozen_cpu_collection_model(
        config,
        vocab_size,
        frozen_state_dict,
    )
    snapshots, train_hit_rate = collect_fn(
        data_path,
        vocab_path,
        collection_model,
        max_node_num,
        model_prob,
        max_examples,
        max_requests,
        train_on_eviction_decision,
    )
    finished_at = time.perf_counter()
    return SimpleNamespace(
        snapshots=snapshots,
        train_hit_rate=train_hit_rate,
        collection_seconds=finished_at - started_at,
        collection_wait_seconds=0.0,
        async_collection=True,
    )


def submit_async_collection(
    executor,
    plan,
    train_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    config: dict,
    vocab_size: int,
    max_node_num: int,
    model_prob: float,
    max_collection_requests,
    train_on_eviction_decision: bool = False,
    collect_fn=collect_snapshots,
):
    frozen_state_dict = freeze_model_state_dict_for_collection(model)
    future = executor.submit(
        collect_snapshots_with_frozen_state,
        dict(config),
        vocab_size,
        frozen_state_dict,
        train_path,
        vocab_path,
        max_node_num,
        model_prob,
        plan.max_examples,
        max_collection_requests,
        train_on_eviction_decision,
        collect_fn,
    )
    return SimpleNamespace(
        future=future,
        plan=plan,
        model_prob=model_prob,
        submitted_at=time.perf_counter(),
    )


def wait_for_async_collection(async_job):
    wait_start = time.perf_counter()
    result = async_job.future.result()
    result.collection_wait_seconds = time.perf_counter() - wait_start
    result.plan = async_job.plan
    result.model_prob = async_job.model_prob
    return result


def collect_round_sync(
    plan,
    train_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    model_prob: float,
    max_collection_requests,
    train_on_eviction_decision: bool = False,
    collect_fn=collect_snapshots,
):
    started_at = time.perf_counter()
    model.eval()
    snapshots, train_hit_rate = collect_fn(
        train_path,
        vocab_path,
        model,
        max_node_num,
        model_prob,
        plan.max_examples,
        max_collection_requests,
        train_on_eviction_decision,
    )
    finished_at = time.perf_counter()
    return SimpleNamespace(
        snapshots=snapshots,
        train_hit_rate=train_hit_rate,
        collection_seconds=finished_at - started_at,
        collection_wait_seconds=0.0,
        async_collection=False,
        plan=plan,
        model_prob=model_prob,
    )


def flatten_eviction_steps(snapshots, max_steps: int = None):
    """Flatten collected training-step containers named eviction_steps."""
    steps = []
    for snapshot in snapshots:
        for step in snapshot.eviction_steps:
            if step.num_candidates >= 2 and getattr(step, "oracle_distances", None) is not None:
                steps.append(step)
                if max_steps is not None and len(steps) >= max_steps:
                    return steps
    return steps


def flatten_microstep_steps(snapshots):
    """Flatten collected microstep training steps in original collection order."""
    return [
        step
        for snapshot in snapshots
        for step in snapshot.eviction_steps
    ]


def count_microstep_steps(snapshots) -> int:
    """Count microstep training steps, not request-level containers."""
    return sum(len(snapshot.eviction_steps) for snapshot in snapshots)


def count_microstep_windows(snapshots, sequence_length: int) -> int:
    steps = count_microstep_steps(snapshots)
    return max(0, steps - sequence_length + 1)


def as_microstep_window_batches(
    snapshots,
    batch_size: int,
    sequence_length: int,
    shuffle: bool = True,
):
    """
    Yield Parrot-style batches of consecutive microstep windows.

    Each yielded sample is a SimpleNamespace(eviction_steps=[...]) containing
    one contiguous window. The batch is a list of these samples, so the existing
    TrieParrotModel.loss(batch) API remains unchanged.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    steps = flatten_microstep_steps(snapshots)
    positions = list(range(max(0, len(steps) - sequence_length + 1)))
    if shuffle:
        random.shuffle(positions)

    for batch_start in range(0, len(positions), batch_size):
        batch = []
        for start in positions[batch_start:batch_start + batch_size]:
            batch.append(SimpleNamespace(
                window_start=start,
                eviction_steps=steps[start:start + sequence_length],
            ))
        if batch:
            yield batch


def collect_rank_eval_steps(
    data_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    max_requests: int,
    max_steps: int = None,
):
    """Collect fixed oracle-policy validation eviction steps for rank diagnostics."""
    snapshots, _ = collect_snapshots(
        data_path,
        vocab_path,
        model,
        max_node_num,
        model_prob=0.0,
        max_examples=None,
        max_requests=max_requests,
    )
    return flatten_eviction_steps(snapshots, max_steps)


def compute_rank_eval_metrics(model: TrieParrotModel, eviction_steps):
    """Evaluate model score ordering against oracle reuse-distance relevances."""
    if not eviction_steps:
        return {
            "rank_eval_ndcg": 0.0,
            "rank_eval_pairwise_acc": 0.0,
            "rank_eval_top1_acc": 0.0,
            "rank_eval_score_std": 0.0,
            "rank_eval_steps": 0,
            "rank_eval_pairs": 0,
        }

    device = next(model.parameters()).device
    ndcgs = []
    score_stds = []
    top1_correct = 0
    pair_correct = 0
    pair_total = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for step in eviction_steps:
            microstep_history_paths = getattr(step, "microstep_history_paths")
            request_history_paths = getattr(step, "request_history_paths")
            microstep_history_memory = model._encode_history_paths(
                microstep_history_paths,
                device,
                max_history=model.max_microstep_history,
            )
            request_history_memory = model._encode_request_history_paths(
                request_history_paths,
                device,
            )
            logits, _ = model(
                microstep_history_memory,
                request_history_memory,
                step.lru_features,
                candidate_paths=step.leaf_paths,
                inference=False,
            )
            scores = logits.squeeze(0).detach().float().cpu().tolist()
            relevances = model._transform_oracle_distances(
                step.oracle_distances,
                list(range(step.num_candidates)),
                device,
            ).detach().float().cpu().tolist()

            n = len(scores)
            if n < 2:
                continue

            score_tensor = torch.tensor(scores, dtype=torch.float32)
            score_stds.append(float(score_tensor.std(unbiased=False).item()))

            best_rel_idx = max(range(n), key=lambda idx: relevances[idx])
            best_score_idx = max(range(n), key=lambda idx: scores[idx])
            if best_rel_idx == best_score_idx:
                top1_correct += 1

            for i in range(n):
                for j in range(i + 1, n):
                    rel_delta = relevances[i] - relevances[j]
                    if rel_delta == 0:
                        continue
                    pair_total += 1
                    score_delta = scores[i] - scores[j]
                    if score_delta * rel_delta > 0:
                        pair_correct += 1

            score_order = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
            ideal_order = sorted(range(n), key=lambda idx: relevances[idx], reverse=True)
            gains = [math.expm1(rel) for rel in relevances]
            dcg = sum(
                gains[idx] / math.log2(rank + 2)
                for rank, idx in enumerate(score_order)
            )
            idcg = sum(
                gains[idx] / math.log2(rank + 2)
                for rank, idx in enumerate(ideal_order)
            )
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    if was_training:
        model.train()

    evaluated_steps = len(ndcgs)
    return {
        "rank_eval_ndcg": sum(ndcgs) / evaluated_steps if evaluated_steps else 0.0,
        "rank_eval_pairwise_acc": pair_correct / pair_total if pair_total else 0.0,
        "rank_eval_top1_acc": top1_correct / evaluated_steps if evaluated_steps else 0.0,
        "rank_eval_score_std": sum(score_stds) / evaluated_steps if evaluated_steps else 0.0,
        "rank_eval_steps": evaluated_steps,
        "rank_eval_pairs": pair_total,
    }


def should_run_periodic_event(step: int, freq: int) -> bool:
    return step > 0 and freq > 0 and step % freq == 0


def round_step_budget(step: int, total_steps: int, dagger_update_freq: int) -> int:
    if step >= total_steps:
        return 0
    return min(max(1, dagger_update_freq), total_steps - step)


def round_collection_examples(
    step: int,
    total_steps: int,
    dagger_update_freq: int,
    batch_size: int,
    collection_multiplier: int = 1,
) -> int:
    safe_multiplier = max(1, int(collection_multiplier or 1))
    return round_step_budget(step, total_steps, dagger_update_freq) * safe_multiplier * batch_size


def resolve_loss_warmup_steps(value, sequence_length: int) -> int:
    """Resolve Parrot-style warmup steps for each training window."""
    if value in (None, "", 0, "0", False):
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"half", "parrot_half"}:
            warmup_steps = sequence_length // 2
        else:
            warmup_steps = int(normalized)
    else:
        warmup_steps = int(value)

    if warmup_steps < 0:
        raise ValueError("loss_warmup_steps must be nonnegative")
    if warmup_steps >= sequence_length:
        raise ValueError(
            "loss_warmup_steps must be smaller than sequence_length, "
            f"got {warmup_steps} for sequence_length={sequence_length}"
        )
    return warmup_steps


def resolve_optional_positive_int(value, field_name: str):
    if value in (None, "", 0, "0", False):
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive when set")
    return resolved


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}")
    if device.index is None:
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    return int(device.index)


def parse_train_device_ids(
    train_devices: str,
    primary_device: torch.device,
    cuda_device_count: int = None,
):
    """Parse a comma-separated CUDA device list for loss parallelism."""
    normalized_train_devices = (
        train_devices.strip().lower()
        if isinstance(train_devices, str)
        else train_devices
    )
    if primary_device.type != "cuda":
        if normalized_train_devices not in (None, "", "none", "single"):
            raise ValueError("--train_devices requires a CUDA --device")
        return []

    if cuda_device_count is None:
        cuda_device_count = torch.cuda.device_count()
    primary_index = cuda_device_index(primary_device)

    if normalized_train_devices in (None, "", "none", "single"):
        device_ids = [primary_index]
    elif normalized_train_devices == "auto":
        device_ids = [primary_index] + [
            idx for idx in range(cuda_device_count)
            if idx != primary_index
        ]
    else:
        device_ids = []
        seen = set()
        for raw_part in normalized_train_devices.split(","):
            part = raw_part.strip().lower()
            if not part:
                continue
            if part.startswith("cuda:"):
                device_id = int(part.split(":", 1)[1])
            elif part.startswith("gpu"):
                device_id = int(part[3:])
            else:
                device_id = int(part)
            if device_id not in seen:
                device_ids.append(device_id)
                seen.add(device_id)

    if not device_ids:
        raise ValueError("--train_devices did not contain any CUDA device ids")
    invalid = [
        device_id for device_id in device_ids
        if device_id < 0 or device_id >= cuda_device_count
    ]
    if invalid:
        raise ValueError(
            f"CUDA device ids out of range for {cuda_device_count} visible GPUs: "
            f"{invalid}"
        )
    return device_ids


def split_batch_for_devices(batch, device_count: int):
    """Split a list batch into non-empty, near-even shards."""
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    if not batch:
        return []

    shard_count = min(device_count, len(batch))
    base = len(batch) // shard_count
    extra = len(batch) % shard_count
    chunks = []
    start = 0
    for shard_idx in range(shard_count):
        size = base + (1 if shard_idx < extra else 0)
        chunks.append(batch[start:start + size])
        start += size
    return chunks


def finalize_loss_stats(stats: dict) -> dict:
    if stats.get("loss_steps", 0):
        stats["hard_lru_active_frac"] = (
            stats.get("hard_lru_cases_count", 0)
            / stats["loss_steps"]
        )
    else:
        stats["hard_lru_active_frac"] = 0.0
    if stats.get("top_set_acc_count", 0):
        stats["top_set_acc"] = (
            stats.get("top_set_acc_correct", 0)
            / stats["top_set_acc_count"]
        )
    else:
        stats["top_set_acc"] = 0.0
    if stats.get("regret_count", 0):
        stats["regret"] = stats.get("regret_sum", 0.0) / stats["regret_count"]
    else:
        stats["regret"] = 0.0

    for prefix in (
        "oracle_target_lcp",
        "lru_target_lcp",
        "model_wrong_target_lcp",
    ):
        count = stats.get(f"{prefix}_count", 0)
        for field in TrieParrotModel._lcp_stat_fields():
            sum_key = f"{prefix}_{field}_sum"
            mean_key = f"{prefix}_{field}_mean"
            stats[mean_key] = (
                stats.get(sum_key, 0.0) / count
                if count
                else 0.0
            )
    return stats


def combine_loss_stats(stats_items) -> dict:
    combined = {}
    derived_keys = {
        "hard_lru_active_frac",
        "top_set_acc",
        "regret",
    }
    for prefix in (
        "oracle_target_lcp",
        "lru_target_lcp",
        "model_wrong_target_lcp",
    ):
        for field in TrieParrotModel._lcp_stat_fields():
            derived_keys.add(f"{prefix}_{field}_mean")

    for stats in stats_items:
        for key, value in stats.items():
            if not isinstance(value, (int, float)):
                continue
            if key in derived_keys:
                continue
            if key == "max_loss_candidates_effective":
                combined[key] = max(combined.get(key, 0), value)
            elif key == "lru_prior_alpha":
                combined[key] = value
            else:
                combined[key] = combined.get(key, 0) + value

    return finalize_loss_stats(combined)


class TrieLossShard(torch.nn.Module):
    """Replica wrapper that returns summed losses plus local loss statistics."""

    def __init__(self, model: TrieParrotModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        snapshots,
        max_candidates,
        max_steps_per_snapshot,
        warmup_steps_per_snapshot,
    ):
        losses = self.model.loss(
            snapshots,
            max_candidates=max_candidates,
            max_steps_per_snapshot=max_steps_per_snapshot,
            warmup_steps_per_snapshot=warmup_steps_per_snapshot,
            reduction="sum",
        )
        return losses, dict(self.model.last_loss_stats)


def merge_loss_shard_outputs(
    outputs,
    primary_device: torch.device,
    normalize: bool = True,
):
    loss_names = TrieParrotModel.loss_names()
    loss_sums = {name: None for name in loss_names}
    counts = {name: 0 for name in loss_names}
    stats_items = []

    for losses, stats in outputs:
        stats_items.append(stats)
        for name in loss_names:
            value = losses[name].to(primary_device)
            loss_sums[name] = value if loss_sums[name] is None else loss_sums[name] + value
            counts[name] += int(stats.get(f"{name}_count", 0))

    combined_stats = combine_loss_stats(stats_items)
    merged_losses = {}
    for name in loss_names:
        if counts[name] > 0:
            if normalize:
                merged_losses[name] = loss_sums[name] / counts[name]
            else:
                merged_losses[name] = loss_sums[name]
        else:
            merged_losses[name] = torch.tensor(
                0.0,
                device=primary_device,
                requires_grad=True,
            )
        combined_stats[f"{name}_count"] = counts[name]

    return merged_losses, combined_stats


def compute_training_losses(
    model: TrieParrotModel,
    batch,
    max_candidates,
    max_steps_per_snapshot,
    warmup_steps_per_snapshot,
    train_device_ids,
    reduction="mean",
):
    """Compute one optimizer-step loss, optionally sharded across CUDA devices."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be one of {'mean', 'sum'}")

    if len(train_device_ids) < 2 or len(batch) < 2:
        return model.loss(
            batch,
            max_candidates=max_candidates,
            max_steps_per_snapshot=max_steps_per_snapshot,
            warmup_steps_per_snapshot=warmup_steps_per_snapshot,
            reduction=reduction,
        )

    primary_device = next(model.parameters()).device
    chunks = split_batch_for_devices(batch, len(train_device_ids))
    active_device_ids = train_device_ids[:len(chunks)]
    if len(active_device_ids) < 2:
        return model.loss(
            batch,
            max_candidates=max_candidates,
            max_steps_per_snapshot=max_steps_per_snapshot,
            warmup_steps_per_snapshot=warmup_steps_per_snapshot,
            reduction=reduction,
        )

    loss_module = TrieLossShard(model)
    replicas = replicate(loss_module, active_device_ids)
    inputs = tuple(
        (
            chunk,
            max_candidates,
            max_steps_per_snapshot,
            warmup_steps_per_snapshot,
        )
        for chunk in chunks
    )
    outputs = parallel_apply(replicas, inputs, devices=active_device_ids)
    losses, stats = merge_loss_shard_outputs(
        outputs,
        primary_device,
        normalize=(reduction == "mean"),
    )
    model.last_loss_stats = stats
    return losses


def synchronize_training_devices_for_timing(device: torch.device, train_device_ids):
    if device.type != "cuda":
        return
    device_ids = train_device_ids or [cuda_device_index(device)]
    for device_id in device_ids:
        torch.cuda.synchronize(device_id)


def iter_loss_microbatches(batch, microbatch_size):
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= len(batch):
        yield batch
        return

    for start in range(0, len(batch), microbatch_size):
        yield batch[start:start + microbatch_size]


def summarize_loss_batch(
    model: TrieParrotModel,
    batch,
    max_candidates,
    max_steps_per_snapshot,
    warmup_steps_per_snapshot,
):
    """Collect loss counts without running the neural scorer."""
    stats = {
        "full_steps": 0,
        "capped_steps": 0,
        "candidate_count": 0,
        "ranking_count": 0,
        "reuse_count": 0,
        "ce_count": 0,
        "top_set_ce_count": 0,
        "hard_lru_margin_count": 0,
        "warmup_steps": 0,
        "loss_steps": 0,
        "microstep_access_steps": 0,
        "eviction_decision_steps": 0,
        "lru_target_kept_count": 0,
        "lru_target_steps": 0,
        "oracle_top_set_kept_count": 0,
        "oracle_top_set_steps": 0,
        "hard_lru_cases_count": 0,
        "hard_lru_active_frac": 0.0,
        "max_loss_candidates_effective": 0,
    }

    for snapshot in batch:
        eviction_steps = snapshot.eviction_steps
        warmup_steps = max(0, int(warmup_steps_per_snapshot or 0))
        if warmup_steps >= len(eviction_steps) and len(eviction_steps) > 0:
            raise ValueError(
                "warmup_steps_per_snapshot must be smaller than the "
                f"number of eviction steps, got {warmup_steps} for "
                f"{len(eviction_steps)} steps"
            )
        if warmup_steps > 0:
            stats["warmup_steps"] += min(warmup_steps, len(eviction_steps))
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
            step_kind = getattr(step, "step_kind", "microstep_access")
            if model._step_kind_loss_weight(step_kind) <= 0.0:
                continue

            stats["loss_steps"] += 1
            if step_kind == "eviction_decision":
                stats["eviction_decision_steps"] += 1
            else:
                stats["microstep_access_steps"] += 1
            oracle_distances = getattr(step, "oracle_distances", None)
            oracle_top_set = tuple(getattr(step, "oracle_top_set", None) or ())
            if not oracle_top_set:
                oracle_top_set = model._oracle_top_set_from_distances(
                    oracle_distances
                )
            lru_target = getattr(step, "lru_target", None)
            required_indices = set(
                getattr(step, "required_candidate_indices", None) or ()
            )
            required_indices.update(oracle_top_set)
            if lru_target is not None:
                required_indices.add(lru_target)
            selected_indices, _ = model._candidate_subset(
                step.num_candidates,
                step.oracle_target,
                max_candidates,
                required_indices,
            )
            selected_set = set(selected_indices)
            if len(selected_indices) == step.num_candidates:
                stats["full_steps"] += 1
            else:
                stats["capped_steps"] += 1
            stats["candidate_count"] += len(selected_indices)
            stats["max_loss_candidates_effective"] = max(
                stats["max_loss_candidates_effective"],
                len(selected_indices),
            )
            if lru_target is not None:
                stats["lru_target_steps"] += 1
                if lru_target in selected_set:
                    stats["lru_target_kept_count"] += 1
            if oracle_top_set:
                stats["oracle_top_set_steps"] += 1
                if set(oracle_top_set).issubset(selected_set):
                    stats["oracle_top_set_kept_count"] += 1
            hard_lru_case = (
                lru_target is not None
                and oracle_top_set
                and lru_target not in set(oracle_top_set)
                and lru_target in selected_set
            )
            if hard_lru_case:
                stats["hard_lru_cases_count"] += 1

            if oracle_distances is not None:
                stats["ranking_count"] += 1
                if model.reuse_loss_weight > 0:
                    stats["reuse_count"] += 1
            if model.ce_loss_weight > 0:
                stats["ce_count"] += 1
            if model.top_set_ce_weight > 0:
                stats["top_set_ce_count"] += 1
            if model.hard_lru_margin_weight > 0 and hard_lru_case:
                stats["hard_lru_margin_count"] += 1

    if stats["loss_steps"] > 0:
        stats["hard_lru_active_frac"] = (
            stats["hard_lru_cases_count"] / stats["loss_steps"]
        )

    return stats


def evaluate(
    data_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    max_requests: int = None,
):
    """Evaluate model hit rate on a dataset (pure model policy)."""
    cache = SequenceTrieCache(
        max_node_num=max_node_num, 
        evict_type=TrieModelPredictAlgorithm, 
        model=model,
    )
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        request_count = 0
        while not trace.done():
            if max_requests is not None and request_count >= max_requests:
                break
            seq = trace.next()
            cache.access(seq)
            request_count += 1
    
    _, hit, miss = cache.stat_info
    total = hit + miss
    return hit / total if total > 0 else 0.0


def evaluate_lru(data_path: str, vocab_path: str, max_node_num: int):
    """Baseline: LRU hit rate."""
    cache = SequenceTrieCache(
        max_node_num=max_node_num, 
        evict_type=TrieLRUAlgorithm,
    )
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        while not trace.done():
            seq = trace.next()
            cache.access(seq)
    
    _, hit, miss = cache.stat_info
    total = hit + miss
    return hit / total if total > 0 else 0.0


def save_training_checkpoint(path: str, model, optimizer, step: int, best_eval_hit_rate: float):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_eval_hit_rate": best_eval_hit_rate,
    }, path)


def load_training_checkpoint(path: str, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_info = model.load_state_dict_compatible(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and not load_info["migrated"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer_state_dict" in checkpoint:
            print(
                "TrieParrot: skipped optimizer state because checkpoint weights "
                "were migrated to deterministic LRU-prior scoring"
            )
        return int(checkpoint.get("step", 0)), float(checkpoint.get("best_eval_hit_rate", 0.0))

    model.load_state_dict_compatible(checkpoint)
    return 0, 0.0


def training_checkpoint_step(path: str) -> int:
    """Parse the optimizer-state checkpoint step from supported filenames."""
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    for prefix in ("training_step_", "training_final_"):
        if stem.startswith(prefix):
            return int(stem[len(prefix):])
    raise ValueError(f"Unsupported training checkpoint name: {name}")


def latest_training_checkpoint(checkpoint_dir: str):
    candidates = (
        glob.glob(os.path.join(checkpoint_dir, "training_step_*.pt"))
        + glob.glob(os.path.join(checkpoint_dir, "training_final_*.pt"))
    )
    if not candidates:
        return None
    return max(candidates, key=training_checkpoint_step)


METRIC_FIELDS = [
    "run_id",
    "event",
    "step",
    "timestamp",
    "loss_total",
    "loss_ranking",
    "loss_reuse",
    "loss_ce",
    "loss_top_set_ce",
    "loss_hard_lru_margin",
    "train_hr",
    "eval_hr",
    "model_prob",
    "full_steps",
    "capped_steps",
    "candidate_count",
    "avg_candidates",
    "max_loss_candidates_effective",
    "microstep_access_steps",
    "eviction_decision_steps",
    "hard_lru_cases_count",
    "hard_lru_active_frac",
    "lru_target_kept_count",
    "lru_target_steps",
    "oracle_top_set_kept_count",
    "oracle_top_set_steps",
    "top_set_acc",
    "regret",
    "lru_prior_alpha",
    "oracle_target_lcp_len_mean",
    "oracle_target_lcp_ratio_candidate_mean",
    "oracle_target_lcp_ratio_current_mean",
    "oracle_target_lcp_candidate_suffix_len_mean",
    "oracle_target_lcp_current_suffix_len_mean",
    "lru_target_lcp_len_mean",
    "lru_target_lcp_ratio_candidate_mean",
    "lru_target_lcp_ratio_current_mean",
    "lru_target_lcp_candidate_suffix_len_mean",
    "lru_target_lcp_current_suffix_len_mean",
    "model_wrong_target_lcp_len_mean",
    "model_wrong_target_lcp_ratio_candidate_mean",
    "model_wrong_target_lcp_ratio_current_mean",
    "model_wrong_target_lcp_candidate_suffix_len_mean",
    "model_wrong_target_lcp_current_suffix_len_mean",
    "num_snapshots",
    "num_microsteps",
    "batch_size",
    "best_eval_hit_rate",
    "checkpoint_path",
    "training_checkpoint_path",
    "rank_eval_ndcg",
    "rank_eval_pairwise_acc",
    "rank_eval_top1_acc",
    "rank_eval_score_std",
    "rank_eval_steps",
    "rank_eval_pairs",
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
]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_metric_fields(metrics_path: str):
    """Upgrade an existing metrics CSV header when new optional fields are added."""
    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        return

    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        if existing_fields == METRIC_FIELDS:
            return
        rows = list(reader)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def append_metric_row(metrics_path: str, row: dict):
    """Append one training metric/event row without truncating existing runs."""
    parent = os.path.dirname(metrics_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0
    if file_exists:
        ensure_metric_fields(metrics_path)

    clean_row = {field: row.get(field, "") for field in METRIC_FIELDS}
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_row)


def plot_loss_curves(metrics_path: str, run_id: str, output_path: str):
    """Plot loss curves for a single run. Plot failures are non-fatal."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: Could not import matplotlib for loss plot: {exc}")
        return False

    rows = []
    try:
        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_id") == run_id and row.get("event") == "train_step":
                    rows.append(row)
    except Exception as exc:
        print(f"WARNING: Could not read metrics for loss plot: {exc}")
        return False

    if not rows:
        print(f"WARNING: No train_step metrics for run_id={run_id}; skipping loss plot")
        return False

    def parse_float(value):
        if value in (None, ""):
            return None
        return float(value)

    try:
        steps = [int(row["step"]) for row in rows]
        series = {
            "loss_total": [parse_float(row.get("loss_total")) for row in rows],
            "loss_ranking": [parse_float(row.get("loss_ranking")) for row in rows],
            "loss_reuse": [parse_float(row.get("loss_reuse")) for row in rows],
            "loss_ce": [parse_float(row.get("loss_ce")) for row in rows],
            "loss_top_set_ce": [
                parse_float(row.get("loss_top_set_ce")) for row in rows
            ],
            "loss_hard_lru_margin": [
                parse_float(row.get("loss_hard_lru_margin")) for row in rows
            ],
        }

        plt.figure(figsize=(10, 6))
        for name, values in series.items():
            if any(value is not None for value in values):
                plt.plot(steps, values, marker="o", label=name)

        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title(f"Trie-PARROT losses run={run_id} steps={steps[0]}..{steps[-1]}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f"WARNING: Could not render loss plot: {exc}")
        return False

    print(f"Loss curves saved to {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrieParrotModel with DAgger')
    parser.add_argument("--dataset", type=str, default='oasst1_timed_global_b16')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("-p", "--model_config_path", type=str, 
                        default='checkpoints/trie_model/model_config.json')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--data_root_dir", type=str, default='data')
    parser.add_argument("--resume_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_auto", action="store_true")
    parser.add_argument(
        "--train_devices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated CUDA device ids for sharding training "
            "loss, e.g. '0,1', 'cuda:4,cuda:5', 'gpu4,gpu5', or 'auto'. "
            "Collection and eval still run on the first device."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        train_device_ids = parse_train_device_ids(args.train_devices, device)
        if cuda_device_index(device) != train_device_ids[0]:
            device = torch.device(f"cuda:{train_device_ids[0]}")
        torch.cuda.set_device(device)
    else:
        train_device_ids = []

    # Load config
    if not os.path.exists(args.model_config_path):
        raise ValueError(f'Config not found: {args.model_config_path}')
    with open(args.model_config_path, 'r') as f:
        config = json.load(f)

    lr = config['lr']
    total_steps = config['total_steps']
    eval_freq = config['eval_freq']
    save_freq = config['save_freq']
    batch_size = config['batch_size']
    sequence_length = int(config.get('sequence_length', 40))
    collection_multiplier = max(1, int(config.get('collection_multiplier', 4) or 1))
    max_collection_requests = config.get('max_collection_requests')
    collection_snapshot_cap = (
        config['max_collection_snapshots']
        if 'max_collection_snapshots' in config
        else 'round_budget'
    )
    max_eval_requests = config.get('max_eval_requests')
    rank_eval_requests = config.get('rank_eval_requests')
    rank_eval_max_steps = config.get('rank_eval_max_steps')
    rank_eval_freq = config.get('rank_eval_freq', eval_freq)
    max_loss_candidates = config.get('max_loss_candidates')
    max_loss_steps_per_snapshot = config.get('max_loss_steps_per_snapshot')
    loss_warmup_steps = resolve_loss_warmup_steps(
        config.get('loss_warmup_steps', 0),
        sequence_length,
    )
    loss_microbatch_size = resolve_optional_positive_int(
        config.get('loss_microbatch_size'),
        "loss_microbatch_size",
    )
    optimizer_steps_per_collection = config.get('optimizer_steps_per_collection')
    shuffle_collected_snapshots = config.get('shuffle_collected_snapshots', False)
    async_collection = resolve_bool_config(config, 'async_collection', False)
    collection_autotune = resolve_bool_config(config, 'collection_autotune', False)
    collection_target_train_time_ratio = config.get(
        'collection_target_train_time_ratio',
        1.0,
    )
    collection_multiplier_min = max(
        1,
        int(config.get('collection_multiplier_min', 1) or 1),
    )
    collection_multiplier_max = resolve_optional_positive_int(
        config.get('collection_multiplier_max'),
        "collection_multiplier_max",
    )
    if collection_multiplier_max is None:
        collection_multiplier_max = (
            max(collection_multiplier, collection_multiplier * 4)
            if collection_autotune
            else collection_multiplier
        )
    collection_multiplier_max = max(collection_multiplier_min, collection_multiplier_max)
    collection_autotune_max_scale = float(
        config.get('collection_autotune_max_scale', 2.0) or 2.0
    )
    max_node_num = config['max_node_num']
    dagger_init = config['dagger_init']
    dagger_final = config['dagger_final']
    dagger_steps = config['dagger_steps']
    dagger_update_freq = config['dagger_update_freq']
    ranking_loss_weight = config.get('ranking_loss_weight', 1.0)
    reuse_loss_weight = config.get('reuse_loss_weight', 0.1)
    ce_loss_weight = config.get('ce_loss_weight', 0.0)
    ce_target_policy = config.get('ce_target_policy', 'argmax')
    top_set_ce_weight = config.get('top_set_ce_weight', 0.0)
    hard_lru_margin_weight = config.get('hard_lru_margin_weight', 0.0)
    hard_lru_margin = config.get('hard_lru_margin', 0.2)
    train_on_eviction_decision = resolve_bool_config(
        config,
        'train_on_eviction_decision',
        False,
    )
    eviction_decision_loss_weight = config.get('eviction_decision_loss_weight', 1.0)
    microstep_access_loss_weight = config.get('microstep_access_loss_weight', 1.0)
    reuse_distance_log_cap = config.get('reuse_distance_log_cap', 5.0)
    ndcg_alpha = config.get('ndcg_alpha', 10.0)
    (
        lru_prior_alpha_init,
        lru_prior_alpha_max,
        lru_prior_alpha_learnable,
    ) = resolve_lru_prior_config(config)

    print(f'TrieParrot: lr={lr}, total_steps={total_steps}, eval_freq={eval_freq}, '
          f'save_freq={save_freq}, batch_size={batch_size}, '
          f'sequence_length={sequence_length}')
    if train_device_ids:
        print(
            "TrieParrot: training loss devices="
            f"{','.join(f'cuda:{device_id}' for device_id in train_device_ids)} "
            f"mode={'multi_gpu' if len(train_device_ids) > 1 else 'single_gpu'}"
        )
    else:
        print(f"TrieParrot: training device={device}")
    candidate_mode = (
        "full" if max_loss_candidates is None else f"capped@{max_loss_candidates}"
    )
    print(
        "TrieParrot: collection/loss caps "
        f"max_collection_requests={max_collection_requests} "
        f"max_collection_microstep_snapshots={collection_snapshot_cap} "
        f"max_eval_requests={max_eval_requests} "
        f"rank_eval_requests={rank_eval_requests} "
        f"rank_eval_max_steps={rank_eval_max_steps} "
        f"rank_eval_freq={rank_eval_freq} "
        f"max_loss_candidates={max_loss_candidates} ({candidate_mode}) "
        f"max_loss_steps_per_snapshot={max_loss_steps_per_snapshot} "
        f"loss_warmup_steps={loss_warmup_steps} "
        f"loss_microbatch_size={loss_microbatch_size} "
        f"optimizer_steps_per_collection={optimizer_steps_per_collection} "
        f"shuffle_collected_snapshots={shuffle_collected_snapshots} "
        f"async_collection={async_collection}"
    )
    print(
        "TrieParrot: collection ratio controls "
        f"collection_multiplier={collection_multiplier} "
        f"collection_autotune={collection_autotune} "
        f"target_train_time_ratio={collection_target_train_time_ratio} "
        f"multiplier_bounds=[{collection_multiplier_min},{collection_multiplier_max}] "
        f"max_scale={collection_autotune_max_scale} "
        f"max_collection_requests={max_collection_requests} "
        f"max_collection_snapshots={collection_snapshot_cap} "
        f"optimizer_steps_per_collection={optimizer_steps_per_collection}; "
        "compare collection_seconds/train_round_seconds"
    )
    print(
        "TrieParrot: loss weights "
        f"ranking={ranking_loss_weight} reuse={reuse_loss_weight} ce={ce_loss_weight} "
        f"top_set_ce={top_set_ce_weight} "
        f"hard_lru_margin={hard_lru_margin_weight}@{hard_lru_margin} "
        f"train_on_eviction_decision={train_on_eviction_decision} "
        f"eviction_decision={eviction_decision_loss_weight} "
        f"microstep_access={microstep_access_loss_weight} "
        f"ce_target_policy={ce_target_policy} "
        f"reuse_distance_log_cap={reuse_distance_log_cap} ndcg_alpha={ndcg_alpha}"
    )
    print(
        "TrieParrot: LRU prior "
        f"alpha_init={lru_prior_alpha_init} "
        f"alpha_max={lru_prior_alpha_max} "
        f"alpha_learnable={lru_prior_alpha_learnable}"
    )
    print(f'TrieParrot: DAgger init={dagger_init}, final={dagger_final}, '
          f'steps={dagger_steps}, update_freq={dagger_update_freq}')
    if async_collection:
        print(
            'TrieParrot: async collection enabled '
            '(thread worker, frozen CPU model snapshots)'
        )
    print(
        'TrieParrot: DAgger round collection uses '
        'round_step_budget * collection_multiplier * batch_size microstep snapshots '
        'when max_collection_snapshots=round_budget; '
        f'collection_multiplier={collection_multiplier}'
    )
    print(f'TrieParrot: max_node_num={max_node_num}')

    # Data paths
    data_dir = os.path.join(args.data_root_dir, args.dataset)
    train_path = os.path.join(data_dir, 'train.pkl')
    valid_path = os.path.join(data_dir, 'valid.pkl')
    test_path = os.path.join(data_dir, 'test.pkl')
    vocab_path = os.path.join(data_dir, 'vocab.json')

    for p in [train_path, vocab_path]:
        if not os.path.exists(p):
            raise ValueError(f'Data file not found: {p}')

    # Read vocab size
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    print(f'TrieParrot: vocab_size={vocab_size}')

    # Override config vocab_size with actual data vocab_size
    config['vocab_size'] = vocab_size
    config['lru_prior_alpha_init'] = lru_prior_alpha_init
    config['lru_prior_alpha_max'] = lru_prior_alpha_max
    config['lru_prior_alpha_learnable'] = lru_prior_alpha_learnable
    config['train_on_eviction_decision'] = train_on_eviction_decision
    config['eviction_decision_loss_weight'] = eviction_decision_loss_weight
    config['microstep_access_loss_weight'] = microstep_access_loss_weight
    config['top_set_ce_weight'] = top_set_ce_weight
    config['hard_lru_margin_weight'] = hard_lru_margin_weight
    config['hard_lru_margin'] = hard_lru_margin
    config['async_collection'] = async_collection
    config['collection_autotune'] = collection_autotune
    config['collection_target_train_time_ratio'] = collection_target_train_time_ratio
    config['collection_multiplier_min'] = collection_multiplier_min
    config['collection_multiplier_max'] = collection_multiplier_max
    config['collection_autotune_max_scale'] = collection_autotune_max_scale
    if train_device_ids:
        config['train_devices'] = [
            f'cuda:{device_id}' for device_id in train_device_ids
        ]

    # Checkpoint dir
    checkpoint_dir = os.path.join(args.checkpoints_root_dir, 'trie_model', args.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = make_run_id()
    metrics_path = os.path.join(checkpoint_dir, 'training_metrics.csv')
    loss_plot_path = os.path.join(checkpoint_dir, f'loss_curves_{run_id}.png')
    print(f'TrieParrot: run_id={run_id}')
    print(f'TrieParrot: metrics_csv={metrics_path}')

    # Save effective config
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Create model
    model = create_trie_parrot_model_from_config(config, vocab_size).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TrieParrot: {total_params} parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Resume after optimizer creation so optimizer state can be restored.
    step = 0
    best_eval_hit_rate = 0.0
    resume_path = args.resume_checkpoint_path
    if args.resume_auto and resume_path is None:
        resume_path = latest_training_checkpoint(checkpoint_dir)
    if resume_path:
        step, best_eval_hit_rate = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            device,
        )
        print(
            f"TrieParrot: resumed from {resume_path} "
            f"at step={step} best_eval_hit_rate={best_eval_hit_rate:.4f}"
        )

    # Baseline
    if os.path.exists(valid_path):
        lru_hit_rate = evaluate_lru(valid_path, vocab_path, max_node_num)
        print(f'Baseline LRU hit rate (valid): {lru_hit_rate:.4f}')

    rank_eval_steps = []
    rank_eval_path = valid_path if os.path.exists(valid_path) else test_path
    if rank_eval_requests:
        rank_eval_steps = collect_rank_eval_steps(
            rank_eval_path,
            vocab_path,
            model,
            max_node_num,
            rank_eval_requests,
            rank_eval_max_steps,
        )
        print(
            f'TrieParrot: rank_eval enabled requests={rank_eval_requests} '
            f'steps={len(rank_eval_steps)} freq={rank_eval_freq}'
        )

    # Training loop
    remaining_steps = max(total_steps - step, 0)
    effective_collection_multiplier = collection_multiplier
    async_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="trie-collection")
        if async_collection
        else None
    )
    pending_collection = None
    with tqdm.tqdm(total=remaining_steps, desc='Training') as pbar:
        postfix = {
            'loss/total': 0.0,
            'loss/ranking': 0.0,
            'loss/reuse': 0.0,
            'loss/ce': 0.0,
            'loss/top': 0.0,
            'loss/hard_lru': 0.0,
            'cand': candidate_mode,
            'train_hr': 0.0,
            'eval_hr': 0.0,
            'model_prob': 0.0,
            'collect_s': 0.0,
            'train_s': 0.0,
            'async': int(async_collection),
        }
        
        while step < total_steps:
            if (
                pending_collection is not None
                and pending_collection.plan.step != step
            ):
                stale_result = wait_for_async_collection(pending_collection)
                print(
                    "\n  WARNING: Discarded async collection for "
                    f"step={pending_collection.plan.step}; current step={step} "
                    f"after training {stale_result.collection_seconds:.2f}s"
                )
                pending_collection = None

            model_prob = get_model_prob(step, dagger_init, dagger_final, dagger_steps)
            postfix['model_prob'] = f'{model_prob:.2f}'
            plan = plan_collection_round(
                step,
                total_steps,
                dagger_update_freq,
                batch_size,
                effective_collection_multiplier,
                collection_snapshot_cap,
            )

            if async_collection:
                if pending_collection is None:
                    pending_collection = submit_async_collection(
                        async_executor,
                        plan,
                        train_path,
                        vocab_path,
                        model,
                        config,
                        vocab_size,
                        max_node_num,
                        model_prob,
                        max_collection_requests,
                        train_on_eviction_decision,
                    )
                collection_result = wait_for_async_collection(pending_collection)
                pending_collection = None
            else:
                collection_result = collect_round_sync(
                    plan,
                    train_path,
                    vocab_path,
                    model,
                    max_node_num,
                    model_prob,
                    max_collection_requests,
                    train_on_eviction_decision,
                )

            plan = collection_result.plan
            model_prob = collection_result.model_prob
            snapshots = collection_result.snapshots
            train_hit_rate = collection_result.train_hit_rate
            round_collection_multiplier = plan.collection_multiplier
            postfix['train_hr'] = f'{train_hit_rate:.4f}'
            postfix['collect_s'] = f'{collection_result.collection_seconds:.1f}'
            
            if not snapshots:
                print('WARNING: No snapshots collected, skipping batch')
                continue

            microstep_count = count_microstep_steps(snapshots)
            window_count = count_microstep_windows(snapshots, sequence_length)
            append_metric_row(metrics_path, {
                "run_id": run_id,
                "event": "collection",
                "step": step,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "train_hr": train_hit_rate,
                "model_prob": model_prob,
                "num_snapshots": microstep_count,
                "num_microsteps": microstep_count,
                "full_steps": window_count,
                "batch_size": batch_size,
                "collection_seconds": collection_result.collection_seconds,
                "collection_wait_seconds": collection_result.collection_wait_seconds,
                "async_collection": int(collection_result.async_collection),
                "collection_multiplier": round_collection_multiplier,
                "collection_autotune": int(collection_autotune),
                "collection_target_train_time_ratio": collection_target_train_time_ratio,
                "max_collection_requests": max_collection_requests,
                "max_collection_snapshots": collection_snapshot_cap,
                "optimizer_steps_per_collection": optimizer_steps_per_collection,
            })
            print(
                f"\n  Collection: step={step} train_hr={train_hit_rate:.4f} "
                f"microsteps={microstep_count} windows={window_count} "
                f"model_prob={model_prob:.2f} "
                f"collect_s={collection_result.collection_seconds:.2f} "
                f"wait_s={collection_result.collection_wait_seconds:.2f} "
                f"async={int(collection_result.async_collection)} "
                f"multiplier={round_collection_multiplier}"
            )
            if window_count == 0:
                print(
                    'WARNING: Not enough microstep snapshots for one training '
                    f'window: collected={microstep_count}, '
                    f'sequence_length={sequence_length}'
                )
                continue

            round_budget = resolve_training_round_budget(
                plan,
                snapshots,
                batch_size,
                sequence_length,
                total_steps,
                optimizer_steps_per_collection,
            )

            if async_collection and round_budget > 0:
                next_step = step + round_budget
                if next_step < total_steps:
                    next_plan = plan_collection_round(
                        next_step,
                        total_steps,
                        dagger_update_freq,
                        batch_size,
                        effective_collection_multiplier,
                        collection_snapshot_cap,
                    )
                    next_model_prob = get_model_prob(
                        next_step,
                        dagger_init,
                        dagger_final,
                        dagger_steps,
                    )
                    pending_collection = submit_async_collection(
                        async_executor,
                        next_plan,
                        train_path,
                        vocab_path,
                        model,
                        config,
                        vocab_size,
                        max_node_num,
                        next_model_prob,
                        max_collection_requests,
                        train_on_eviction_decision,
                    )
                    print(
                        f"  Async collection queued: step={next_step} "
                        f"model_prob={next_model_prob:.2f} "
                        f"max_examples={next_plan.max_examples} "
                        f"multiplier={next_plan.collection_multiplier}"
                    )

            synchronize_training_devices_for_timing(device, train_device_ids)
            train_round_started_at = time.perf_counter()

            # Train on Parrot-style consecutive microstep windows.
            model.train()
            round_steps = 0
            for batch in as_microstep_window_batches(
                snapshots,
                batch_size,
                sequence_length,
                shuffle=shuffle_collected_snapshots,
            ):
                if step >= total_steps:
                    break
                if round_steps >= round_budget:
                    break

                # Eval
                if should_run_periodic_event(step, eval_freq):
                    model.eval()
                    if os.path.exists(valid_path):
                        eval_hit_rate = evaluate(
                            valid_path,
                            vocab_path,
                            model,
                            max_node_num,
                            max_eval_requests,
                        )
                    else:
                        eval_hit_rate = evaluate(
                            test_path,
                            vocab_path,
                            model,
                            max_node_num,
                            max_eval_requests,
                        )
                    postfix['eval_hr'] = f'{eval_hit_rate:.4f}'
                    
                    if eval_hit_rate > best_eval_hit_rate:
                        best_eval_hit_rate = eval_hit_rate
                        best_path = os.path.join(checkpoint_dir, 'best.ckpt')
                        torch.save(model.state_dict(), best_path)
                        append_metric_row(metrics_path, {
                            "run_id": run_id,
                            "event": "eval",
                            "step": step,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "eval_hr": eval_hit_rate,
                            "best_eval_hit_rate": best_eval_hit_rate,
                            "checkpoint_path": best_path,
                        })
                        print(f'\n  New best: {eval_hit_rate:.4f}, saved to {best_path}')
                    else:
                        append_metric_row(metrics_path, {
                            "run_id": run_id,
                            "event": "eval",
                            "step": step,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "eval_hr": eval_hit_rate,
                            "best_eval_hit_rate": best_eval_hit_rate,
                        })
                    model.train()

                if (
                    rank_eval_steps
                    and should_run_periodic_event(step, rank_eval_freq)
                ):
                    model.eval()
                    rank_metrics = compute_rank_eval_metrics(model, rank_eval_steps)
                    append_metric_row(metrics_path, {
                        "run_id": run_id,
                        "event": "rank_eval",
                        "step": step,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model_prob": model_prob,
                        "best_eval_hit_rate": best_eval_hit_rate,
                        **rank_metrics,
                    })
                    model.train()
                
                # Save checkpoint
                if should_run_periodic_event(step, save_freq):
                    save_path = os.path.join(checkpoint_dir, f'step_{step}.ckpt')
                    torch.save(model.state_dict(), save_path)
                    training_state_path = os.path.join(
                        checkpoint_dir,
                        f'training_step_{step}.pt',
                    )
                    save_training_checkpoint(
                        training_state_path,
                        model,
                        optimizer,
                        step,
                        best_eval_hit_rate,
                    )
                    append_metric_row(metrics_path, {
                        "run_id": run_id,
                        "event": "checkpoint",
                        "step": step,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "best_eval_hit_rate": best_eval_hit_rate,
                        "checkpoint_path": save_path,
                        "training_checkpoint_path": training_state_path,
                    })
                    print(
                        f'\n  Checkpoint saved: {save_path}; '
                        f'training state: {training_state_path}'
                    )
                
                # Forward + backward
                optimizer.zero_grad(set_to_none=True)

                use_loss_microbatch = (
                    loss_microbatch_size is not None
                    and loss_microbatch_size < len(batch)
                )
                if use_loss_microbatch:
                    loss_stats = summarize_loss_batch(
                        model,
                        batch,
                        max_loss_candidates,
                        max_loss_steps_per_snapshot,
                        loss_warmup_steps,
                    )
                    loss_names = TrieParrotModel.loss_names()
                    loss_values = {name: 0.0 for name in loss_names}
                    total_loss_value = 0.0
                    microbatch_stats = []

                    for micro_batch in iter_loss_microbatches(
                        batch,
                        loss_microbatch_size,
                    ):
                        losses = compute_training_losses(
                            model,
                            micro_batch,
                            max_loss_candidates,
                            max_loss_steps_per_snapshot,
                            loss_warmup_steps,
                            train_device_ids,
                            reduction="sum",
                        )
                        microbatch_stats.append(dict(model.last_loss_stats))
                        scaled_terms = []
                        for name in loss_names:
                            value = losses[name]
                            count = int(loss_stats.get(f"{name}_count", 0))
                            scaled = value / count if count > 0 else value * 0.0
                            scaled_terms.append(scaled)
                            loss_values[name] += float(scaled.detach().item())

                        micro_total_loss = torch.stack(scaled_terms).sum()
                        if not torch.isfinite(micro_total_loss):
                            raise RuntimeError(
                                "Non-finite training loss at "
                                f"step={step}: "
                                + ", ".join(
                                    f"{name}={value.detach().item()}"
                                    for name, value in losses.items()
                                )
                            )
                        total_loss_value += float(micro_total_loss.detach().item())
                        micro_total_loss.backward()
                        del micro_total_loss
                        del scaled_terms
                        del losses
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    model.last_loss_stats = combine_loss_stats(microbatch_stats)
                    optimizer.step()
                else:
                    losses = compute_training_losses(
                        model,
                        batch,
                        max_loss_candidates,
                        max_loss_steps_per_snapshot,
                        loss_warmup_steps,
                        train_device_ids,
                    )
                    total_loss = sum(losses.values())
                    if not torch.isfinite(total_loss):
                        raise RuntimeError(
                            "Non-finite training loss at "
                            f"step={step}: "
                            + ", ".join(
                                f"{name}={value.item()}"
                                for name, value in losses.items()
                            )
                        )
                    total_loss.backward()
                    optimizer.step()
                    loss_values = {
                        name: float(value.detach().item())
                        for name, value in losses.items()
                    }
                    total_loss_value = float(total_loss.detach().item())

                loss_stats = getattr(model, 'last_loss_stats', {})
                full_steps = int(loss_stats.get('full_steps', 0))
                capped_steps = int(loss_stats.get('capped_steps', 0))
                candidate_count = int(loss_stats.get('candidate_count', 0))
                step_count = full_steps + capped_steps
                avg_candidates = candidate_count / step_count if step_count else 0.0

                postfix['loss/total'] = f'{total_loss_value:.4f}'
                postfix['loss/ranking'] = f'{loss_values.get("ranking", 0.0):.4f}'
                postfix['loss/reuse'] = f'{loss_values.get("reuse", 0.0):.4f}'
                postfix['loss/ce'] = f'{loss_values.get("ce", 0.0):.4f}'
                postfix['loss/top'] = f'{loss_values.get("top_set_ce", 0.0):.4f}'
                postfix['loss/hard_lru'] = (
                    f'{loss_values.get("hard_lru_margin", 0.0):.4f}'
                )
                postfix['cand'] = (
                    f'full:{full_steps}/cap:{capped_steps}/avg:{avg_candidates:.1f}'
                )
                append_metric_row(metrics_path, {
                    "run_id": run_id,
                    "event": "train_step",
                    "step": step,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "loss_total": total_loss_value,
                    "loss_ranking": loss_values.get("ranking", 0.0),
                    "loss_reuse": loss_values.get("reuse", 0.0),
                    "loss_ce": loss_values.get("ce", 0.0),
                    "loss_top_set_ce": loss_values.get("top_set_ce", 0.0),
                    "loss_hard_lru_margin": loss_values.get("hard_lru_margin", 0.0),
                    "train_hr": train_hit_rate,
                    "eval_hr": postfix.get('eval_hr', ""),
                    "model_prob": model_prob,
                    "full_steps": full_steps,
                    "capped_steps": capped_steps,
                    "candidate_count": candidate_count,
                    "avg_candidates": avg_candidates,
                    "max_loss_candidates_effective": loss_stats.get(
                        "max_loss_candidates_effective",
                        "",
                    ),
                    "microstep_access_steps": loss_stats.get(
                        "microstep_access_steps",
                        "",
                    ),
                    "eviction_decision_steps": loss_stats.get(
                        "eviction_decision_steps",
                        "",
                    ),
                    "hard_lru_cases_count": loss_stats.get(
                        "hard_lru_cases_count",
                        "",
                    ),
                    "hard_lru_active_frac": loss_stats.get(
                        "hard_lru_active_frac",
                        "",
                    ),
                    "lru_target_kept_count": loss_stats.get(
                        "lru_target_kept_count",
                        "",
                    ),
                    "lru_target_steps": loss_stats.get("lru_target_steps", ""),
                    "oracle_top_set_kept_count": loss_stats.get(
                        "oracle_top_set_kept_count",
                        "",
                    ),
                    "oracle_top_set_steps": loss_stats.get(
                        "oracle_top_set_steps",
                        "",
                    ),
                    "top_set_acc": loss_stats.get("top_set_acc", ""),
                    "regret": loss_stats.get("regret", ""),
                    "lru_prior_alpha": float(model.lru_prior_alpha().detach().item()),
                    "oracle_target_lcp_len_mean": loss_stats.get(
                        "oracle_target_lcp_lcp_len_mean",
                        "",
                    ),
                    "oracle_target_lcp_ratio_candidate_mean": loss_stats.get(
                        "oracle_target_lcp_lcp_ratio_candidate_mean",
                        "",
                    ),
                    "oracle_target_lcp_ratio_current_mean": loss_stats.get(
                        "oracle_target_lcp_lcp_ratio_current_mean",
                        "",
                    ),
                    "oracle_target_lcp_candidate_suffix_len_mean": loss_stats.get(
                        "oracle_target_lcp_candidate_suffix_len_mean",
                        "",
                    ),
                    "oracle_target_lcp_current_suffix_len_mean": loss_stats.get(
                        "oracle_target_lcp_current_suffix_len_mean",
                        "",
                    ),
                    "lru_target_lcp_len_mean": loss_stats.get(
                        "lru_target_lcp_lcp_len_mean",
                        "",
                    ),
                    "lru_target_lcp_ratio_candidate_mean": loss_stats.get(
                        "lru_target_lcp_lcp_ratio_candidate_mean",
                        "",
                    ),
                    "lru_target_lcp_ratio_current_mean": loss_stats.get(
                        "lru_target_lcp_lcp_ratio_current_mean",
                        "",
                    ),
                    "lru_target_lcp_candidate_suffix_len_mean": loss_stats.get(
                        "lru_target_lcp_candidate_suffix_len_mean",
                        "",
                    ),
                    "lru_target_lcp_current_suffix_len_mean": loss_stats.get(
                        "lru_target_lcp_current_suffix_len_mean",
                        "",
                    ),
                    "model_wrong_target_lcp_len_mean": loss_stats.get(
                        "model_wrong_target_lcp_lcp_len_mean",
                        "",
                    ),
                    "model_wrong_target_lcp_ratio_candidate_mean": loss_stats.get(
                        "model_wrong_target_lcp_lcp_ratio_candidate_mean",
                        "",
                    ),
                    "model_wrong_target_lcp_ratio_current_mean": loss_stats.get(
                        "model_wrong_target_lcp_lcp_ratio_current_mean",
                        "",
                    ),
                    "model_wrong_target_lcp_candidate_suffix_len_mean": loss_stats.get(
                        "model_wrong_target_lcp_candidate_suffix_len_mean",
                        "",
                    ),
                    "model_wrong_target_lcp_current_suffix_len_mean": loss_stats.get(
                        "model_wrong_target_lcp_current_suffix_len_mean",
                        "",
                    ),
                    "num_snapshots": microstep_count,
                    "num_microsteps": microstep_count,
                    "batch_size": len(batch),
                    "best_eval_hit_rate": best_eval_hit_rate,
                    "collection_seconds": collection_result.collection_seconds,
                    "collection_wait_seconds": collection_result.collection_wait_seconds,
                    "async_collection": int(collection_result.async_collection),
                    "collection_multiplier": round_collection_multiplier,
                    "collection_autotune": int(collection_autotune),
                    "collection_target_train_time_ratio": collection_target_train_time_ratio,
                    "max_collection_requests": max_collection_requests,
                    "max_collection_snapshots": collection_snapshot_cap,
                    "optimizer_steps_per_collection": optimizer_steps_per_collection,
                })
                pbar.set_postfix(postfix)
                pbar.update(1)
                step += 1
                round_steps += 1
                if "total_loss" in locals():
                    del total_loss
                if "losses" in locals():
                    del losses
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                if step >= total_steps:
                    break

            synchronize_training_devices_for_timing(device, train_device_ids)
            train_round_seconds = time.perf_counter() - train_round_started_at
            collection_train_time_ratio = (
                collection_result.collection_seconds / train_round_seconds
                if train_round_seconds > 0
                else 0.0
            )
            postfix['train_s'] = f'{train_round_seconds:.1f}'
            pbar.set_postfix(postfix)
            append_metric_row(metrics_path, {
                "run_id": run_id,
                "event": "train_round",
                "step": step,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "train_hr": train_hit_rate,
                "model_prob": model_prob,
                "num_snapshots": microstep_count,
                "num_microsteps": microstep_count,
                "full_steps": window_count,
                "batch_size": batch_size,
                "collection_seconds": collection_result.collection_seconds,
                "collection_wait_seconds": collection_result.collection_wait_seconds,
                "train_round_seconds": train_round_seconds,
                "collection_train_time_ratio": collection_train_time_ratio,
                "async_collection": int(collection_result.async_collection),
                "collection_multiplier": round_collection_multiplier,
                "collection_autotune": int(collection_autotune),
                "collection_target_train_time_ratio": collection_target_train_time_ratio,
                "max_collection_requests": max_collection_requests,
                "max_collection_snapshots": collection_snapshot_cap,
                "optimizer_steps_per_collection": optimizer_steps_per_collection,
            })
            print(
                f"  Train round: step={step} trained_steps={round_steps} "
                f"train_s={train_round_seconds:.2f} "
                f"collect/train={collection_train_time_ratio:.2f} "
                f"async={int(collection_result.async_collection)}"
            )
            if collection_autotune:
                tuned_multiplier = tune_collection_multiplier(
                    effective_collection_multiplier,
                    collection_result.collection_seconds,
                    train_round_seconds,
                    collection_target_train_time_ratio,
                    collection_multiplier_min,
                    collection_multiplier_max,
                    collection_autotune_max_scale,
                )
                if tuned_multiplier != effective_collection_multiplier:
                    print(
                        "  Collection autotune: "
                        f"multiplier {effective_collection_multiplier} -> "
                        f"{tuned_multiplier} "
                        f"(target_ratio={collection_target_train_time_ratio})"
                    )
                    effective_collection_multiplier = tuned_multiplier
    
    if pending_collection is not None:
        pending_collection.future.cancel()
    if async_executor is not None:
        async_executor.shutdown(wait=True)

    # Final save
    final_path = os.path.join(checkpoint_dir, f'final_{step}.ckpt')
    torch.save(model.state_dict(), final_path)
    final_training_path = os.path.join(checkpoint_dir, f'training_final_{step}.pt')
    save_training_checkpoint(
        final_training_path,
        model,
        optimizer,
        step,
        best_eval_hit_rate,
    )
    append_metric_row(metrics_path, {
        "run_id": run_id,
        "event": "final",
        "step": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_eval_hit_rate": best_eval_hit_rate,
        "checkpoint_path": final_path,
        "training_checkpoint_path": final_training_path,
    })
    plot_loss_curves(metrics_path, run_id, loss_plot_path)
    print(f'Training complete. Final checkpoint: {final_path}')
    print(f'Best eval hit rate: {best_eval_hit_rate:.4f}')
