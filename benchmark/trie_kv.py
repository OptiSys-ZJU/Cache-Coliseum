import argparse
import csv
import json
import os
import pickle
import random
from functools import partial
from typing import Dict, Iterable, List

from cache.trie.oracle import PrefixFutureOracle
from cache.trie.trie_algorithms import (
    TrieLRUAlgorithm,
    TrieModelGuard,
    TrieModelPredictAlgorithm,
    TrieOracleAlgorithm,
    TrieRandAlgorithm,
)
from cache.trie.trie_cache import SequenceTrieCache


SUM_KEYS = (
    "requests",
    "request_full_hits",
    "prefix_hit_sum",
    "total_blocks",
    "hit_blocks",
    "miss_blocks",
    "recompute_blocks",
    "saved_prefill_tokens",
    "uncacheable_blocks",
    "evictions",
)


def load_data(data_dir: str, split: str, max_requests: int = None):
    data_path = os.path.join(data_dir, f"{split}.pkl")
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)

    metadata = {}
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    if max_requests is not None:
        sequences = sequences[:max_requests]

    return sequences, metadata


def split_by_trace(
    sequences: List[List[int]],
    metadata: dict,
    split: str,
    max_requests: int = None,
):
    split_meta = metadata.get("splits", {}).get(split, {})
    counts = split_meta.get("trace_request_counts")
    if not counts:
        return [sequences]

    chunks = []
    cursor = 0
    remaining = max_requests
    for count in counts:
        if remaining is not None and remaining <= 0:
            break
        take = count if remaining is None else min(count, remaining)
        chunk = sequences[cursor:cursor + take]
        if chunk:
            chunks.append(chunk)
        cursor += count
        if remaining is not None:
            remaining -= take
    return chunks


def interleave_trace_chunks(
    trace_chunks: Iterable[List[List[int]]],
    seed: int = 0,
):
    traces = [list(chunk) for chunk in trace_chunks if chunk]
    rng = random.Random(seed)
    rng.shuffle(traces)

    positions = [0] * len(traces)
    active = list(range(len(traces)))
    interleaved = []
    cursor = 0

    while active:
        if cursor >= len(active):
            cursor = 0

        trace_idx = active[cursor]
        interleaved.append(traces[trace_idx][positions[trace_idx]])
        positions[trace_idx] += 1

        if positions[trace_idx] >= len(traces[trace_idx]):
            active.pop(cursor)
        else:
            cursor += 1

    return interleaved


def load_model(config_path: str, checkpoint_path: str, device: str = "cpu"):
    if not os.path.exists(config_path):
        raise ValueError(f"Model config not found: {config_path}")
    if not checkpoint_path:
        raise ValueError("--model_checkpoint_path is required for model/guard policies")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("Model policies require torch") from exc

    from model.trie_model.model import TrieParrotModel

    model = TrieParrotModel.from_config(config_path, checkpoint_path)
    model.to(torch.device(device))
    model.eval()
    return model


def make_cache(
    policy: str,
    capacity: int,
    sequences: List[List[int]],
    model=None,
    variance_threshold: float = 0.01,
):
    if policy == "lru":
        return SequenceTrieCache(capacity, evict_type=TrieLRUAlgorithm)
    if policy == "rand":
        return SequenceTrieCache(capacity, evict_type=TrieRandAlgorithm)
    if policy == "oracle":
        future_oracle = PrefixFutureOracle(sequences, max_prefix_len=capacity)
        evict_type = partial(TrieOracleAlgorithm, future_oracle=future_oracle)
        return SequenceTrieCache(capacity, evict_type=evict_type)
    if policy == "model":
        return SequenceTrieCache(
            capacity,
            evict_type=TrieModelPredictAlgorithm,
            model=model,
        )
    if policy == "guard":
        return SequenceTrieCache(
            capacity,
            evict_type=TrieModelGuard,
            model=model,
            variance_threshold=variance_threshold,
        )
    raise ValueError(f"Unknown policy: {policy}")


def finalize_stats(stats: Dict[str, float], chunks: int = 1):
    requests = stats.get("requests", 0)
    total_blocks = stats.get("total_blocks", 0)

    stats["block_hit_rate"] = (
        stats.get("hit_blocks", 0) / total_blocks
        if total_blocks
        else 0.0
    )
    stats["request_full_hit_rate"] = (
        stats.get("request_full_hits", 0) / requests
        if requests
        else 0.0
    )
    stats["avg_prefix_hit_len"] = (
        stats.get("prefix_hit_sum", 0) / requests
        if requests
        else 0.0
    )
    stats["chunks"] = chunks
    return stats


def run_policy(
    policy: str,
    capacity: int,
    chunks: Iterable[List[List[int]]],
    block_size: int,
    model=None,
    variance_threshold: float = 0.01,
):
    merged = {key: 0 for key in SUM_KEYS}
    resident_sum = 0
    guard_evictions = 0
    total_guard_evictions = 0
    chunk_count = 0

    for sequences in chunks:
        chunk_count += 1
        cache = make_cache(
            policy,
            capacity,
            sequences,
            model=model,
            variance_threshold=variance_threshold,
        )
        for sequence in sequences:
            cache.access(sequence)

        stats = cache.kv_stat(block_size=block_size)
        for key in SUM_KEYS:
            merged[key] += stats.get(key, 0)
        resident_sum += stats.get("resident_blocks", 0)

        alg = cache.alg
        if hasattr(alg, "guarded_evictions"):
            guard_evictions += alg.guarded_evictions
            total_guard_evictions += alg.total_evictions

    merged["avg_resident_blocks"] = resident_sum / chunk_count if chunk_count else 0
    if total_guard_evictions:
        merged["guard_rate"] = guard_evictions / total_guard_evictions
    else:
        merged["guard_rate"] = 0.0
    return finalize_stats(merged, chunks=chunk_count)


def print_rows(rows):
    header = [
        "policy",
        "capacity",
        "block_token_size",
        "capacity_tokens",
        "requests",
        "block_hit_rate",
        "request_full_hit_rate",
        "avg_prefix_hit_len",
        "recompute_blocks",
        "evictions",
        "guard_rate",
    ]
    print(",".join(header))
    for row in rows:
        print(",".join(str(row.get(key, "")) for key in header))


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefix-trie KV cache policies")
    parser.add_argument("--dataset", type=str, default="oasst1_timed_global_b16")
    parser.add_argument("--data_root_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--capacity", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--policy", nargs="+", default=["lru", "rand", "oracle"],
                        choices=["lru", "rand", "oracle", "model", "guard"])
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--reset_per_trace", action="store_true")
    parser.add_argument("--interleave_traces", action="store_true")
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = os.path.join(args.data_root_dir, args.dataset)
    sequences, metadata = load_data(data_dir, args.split, args.max_requests)
    block_size = args.block_size or metadata.get("block_size", 1)

    if args.reset_per_trace and args.interleave_traces:
        raise ValueError("--reset_per_trace and --interleave_traces are mutually exclusive")

    if args.interleave_traces:
        trace_chunks = split_by_trace(sequences, metadata, args.split, args.max_requests)
        chunks = [interleave_trace_chunks(trace_chunks, seed=args.seed)]
        trace_mode = "interleaved_shared"
    elif args.reset_per_trace:
        chunks = split_by_trace(sequences, metadata, args.split, args.max_requests)
        trace_mode = "reset_per_trace"
    else:
        chunks = [sequences]
        if metadata.get("event_order") == "timestamp":
            trace_mode = "timestamp_shared"
        else:
            trace_mode = "contiguous_shared"

    needs_model = any(policy in ("model", "guard") for policy in args.policy)
    model = None
    if needs_model:
        if not args.model_config_path:
            raise ValueError("--model_config_path is required for model/guard policies")
        model = load_model(
            args.model_config_path,
            args.model_checkpoint_path,
            args.device,
        )

    rows = []
    for capacity in args.capacity:
        for policy in args.policy:
            stats = run_policy(
                policy,
                capacity,
                chunks,
                block_size=block_size,
                model=model,
                variance_threshold=args.variance_threshold,
            )
            row = {
                "dataset": args.dataset,
                "split": args.split,
                "trace_mode": trace_mode,
                "block_token_size": block_size,
                "policy": policy,
                "capacity": capacity,
                "capacity_tokens": capacity * block_size,
                **stats,
            }
            rows.append(row)

    print_rows(rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
