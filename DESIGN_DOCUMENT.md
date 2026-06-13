# Learning-Augmented Prefix-Tree KV Cache Design

This document captures the current `trie-cache` branch state for continuing
work on LLM prefix/KV-cache simulation after cloning the repository on another
machine.

## Current Handoff

The active OASST1 timed shared-cache experiment state is tracked in
`docs/oasst1_timed_experiment_manifest.md`. Read that manifest first before
continuing experiments or training; it records the current workload semantics,
generated files, completed block-size/capacity sweeps, missing settings, and
recommended model-training targets.

## Goal

Build a prefix-tree cache simulator for multi-turn LLM serving traces where each
request is represented by a sequence of KV block hashes.

The target research direction is:

```text
Learning-Augmented Prefix-Tree KV Cache Eviction for Multi-turn LLM Serving
```

The current implementation focuses on a reliable offline simulator first:

```text
OASST1 timed request events
        -> dense global KV block sequences
        -> prefix trie KV cache
        -> LRU / Random / Oracle / Model / Guard eviction
        -> KV-oriented metrics and capacity sweep
```

## Concept Mapping

| LLM serving concept | Repository abstraction |
| --- | --- |
| Request prompt KV blocks | `List[int]` sequence |
| KV block hash id | Trie edge / node id |
| Cached prompt prefix | Root-to-node path in trie |
| GPU KV cache budget | `max_node_num` / benchmark capacity |
| Prefix cache hit | Longest matched trie prefix |
| Prefill recompute | Unmatched suffix blocks |
| Eviction | Delete leaf blocks from prefix trie |
| Belady next use | Next request whose sequence has a candidate path as prefix |

## Important Files

- `cache/trie/oracle.py`
  - Defines `PrefixFutureOracle`.
  - Builds `prefix_tuple -> deque[future_request_indices]`.
  - Replaces repeated future trace scans with O(1) queue-head lookups.

- `cache/trie/trie_algorithms.py`
  - Fixes `TrieNode.get_path_tuple_from_node()`.
  - Adds `eviction_count` accounting.
  - Adds `TrieOracleAlgorithm`, a Belady-style prefix-trie oracle.
  - Keeps torch optional so LRU/Random/Oracle simulation can run without model
    dependencies.

- `cache/trie/trie_cache.py`
  - Extends `SequenceTrieCache` with KV metrics.
  - Handles requests longer than capacity by caching only the first
    `max_node_num` blocks and counting the rest as misses.
  - Updates `TrieTrainingCache` to use `PrefixFutureOracle` for oracle labels.

- `scripts/data_process/preprocess_oasst1_timed.py`
  - Converts timestamped OASST1 conversation events into
    `SequenceTrieDataTrace`-compatible pkl files.
  - Supports session-scoped or global KV block identity.
  - Outputs split pkl files, split event JSONL files, `vocab.json`, and
    `metadata.json`.

- `benchmark/trie_kv.py`
  - Runs capacity sweeps for `lru`, `rand`, `oracle`, `model`, and `guard`.
  - Reports KV-cache metrics such as block hit rate, request full hit rate,
    average prefix hit length, recompute blocks, saved prefill tokens, and
    evictions.

- `tests/test_prefix_oracle.py`
  - Covers prefix future lookup, path recovery, long-request handling, and the
    oracle cache path.

## KV Block to Dense Id Mapping

OASST1 timed preprocessing turns each conversation event into fixed-size KV
block identities. The trie simulator can use raw identities as dict keys, but
`TrieParrotModel` uses `nn.Embedding`, which requires compact integer ids in
`[0, vocab_size)`.

The preprocessing step maps each raw block/token-chunk identity to a stable
dense integer:

```text
raw block identity      dense id
("global", "...")  ->   1
("global", "...")  ->   2
```

Id `0` is reserved for unknown/padding. The simulator should not collapse
unknown KV blocks to id `0`, because that would create false prefix hits. The
active OASST1 timed main line uses global identity scope so shared cache
experiments can observe reuse across timestamp-ordered requests.

## Prefix Future Oracle

For Belady-style eviction, a cached leaf path is reused when a future request
starts with that path.

Example requests:

```python
0: [A, B, C]
1: [A, B, D]
2: [X]
3: [A, B, C]
```

The oracle index stores:

```text
(A)       -> deque([0, 1, 3])
(A, B)    -> deque([0, 1, 3])
(A, B, C) -> deque([0, 3])
(A, B, D) -> deque([1])
(X)       -> deque([2])
```

At request `i`, the current request is consumed from all of its prefix queues.
The next use of a cached leaf path is then the first item in the corresponding
deque. Empty deque means the path is never reused again, so its next use is
`inf` and it is the best eviction candidate.

## Long Request Policy

If a request has more blocks than cache capacity, the current simulator:

```text
cacheable prefix = sequence[:capacity]
uncacheable suffix = sequence[capacity:]
```

Only the cacheable prefix participates in trie insertion/eviction. The full
request length is still used for metrics, so the uncacheable suffix contributes
to misses and recompute blocks.

This is a conservative first policy. Future work can compare alternatives such
as suffix bypass, sliding-window caching, or partial protected-path eviction.

## Preprocessing

Build the current OASST1 timed global dataset:

```bash
python scripts/data_process/preprocess_oasst1_timed.py \
  --output_dir data/oasst1_timed_global_b16 \
  --block_token_size 16 \
  --identity_scope global \
  --event_role all
```

For a validation-only smoke pass:

```bash
python scripts/data_process/preprocess_oasst1_timed.py \
  --output_dir data/oasst1_timed_global_b16 \
  --source_splits validation \
  --block_token_size 16 \
  --identity_scope global \
  --event_role all
```

The script expects the OASST1 sequence/request JSON and CSV files under
`data/traces` by default.

## Benchmarking

Run non-model baselines:

```bash
python -m benchmark.trie_kv \
  --dataset oasst1_timed_global_b16 \
  --split valid \
  --capacity 256 512 1024 2048 4096 \
  --policy lru rand oracle
```

Write results to CSV:

```bash
python -m benchmark.trie_kv \
  --dataset oasst1_timed_global_b16 \
  --split valid \
  --capacity 256 512 1024 2048 4096 \
  --policy lru rand oracle \
  --output_csv res/oasst1_timed_global_b16_valid_kv.csv
```

Run model/guard policies after training:

```bash
python -m benchmark.trie_kv \
  --dataset oasst1_timed_global_b16 \
  --split valid \
  --capacity 256 \
  --policy model guard \
  --model_config_path checkpoints/trie_model/oasst1_timed_global_b16/config.json \
  --model_checkpoint_path checkpoints/trie_model/oasst1_timed_global_b16/best.ckpt \
  --device cpu
```

## Model Training

The existing training entrypoint can consume the preprocessed dataset because it
uses `SequenceTrieDataTrace`.

Example:

```bash
# Create this config first if it does not exist locally.
python -m model.trie_model \
  --dataset oasst1_timed_global_b16 \
  --device cpu \
  --data_root_dir data \
  --model_config_path checkpoints/trie_model/model_config.json
```

For real model training/evaluation, install a compatible torch environment.
The simulator paths for `lru`, `rand`, and `oracle` intentionally do not require
torch.

## Metrics

`SequenceTrieCache.kv_stat()` and `benchmark/trie_kv.py` report:

- `requests`
- `total_blocks`
- `hit_blocks`
- `miss_blocks`
- `block_hit_rate`
- `request_full_hit_rate`
- `avg_prefix_hit_len`
- `recompute_blocks`
- `saved_prefill_tokens`
- `uncacheable_blocks`
- `evictions`
- `resident_blocks` / `avg_resident_blocks`
- `guard_rate` for guard policy

## Validation Commands

These commands passed in the current development environment:

```bash
python tests/test_prefix_oracle.py
PYTHONPATH=. python tests/test_seq_cache.py
PYTHONPATH=. python tests/test_evict.py
PYTHONPATH=. python tests/test_training_cache.py
python -m py_compile \
  cache/trie/oracle.py \
  cache/trie/trie_algorithms.py \
  cache/trie/trie_cache.py \
  scripts/data_process/preprocess_oasst1_timed.py \
  benchmark/trie_kv.py \
  tests/test_prefix_oracle.py
```

## Current Limitations

- OASST1 timed preprocessing depends on the local files under `data/traces`.
- Torch is not installed in the current environment, so model/guard inference
  with a real model checkpoint was not executed locally.
- `TrieModelPredictAlgorithm` still updates history with only the last block of
  each request. This is acceptable for the first wiring pass but should be
  revisited for KV traces.
- The long-request policy currently caches only the prefix up to capacity.
- The oracle currently optimizes block hit behavior under leaf eviction, not
  heterogeneous CPU/SSD/GPU transfer costs.

## Suggested Next Steps

1. Use `docs/oasst1_timed_experiment_manifest.md` as the current experiment
   handoff.
2. Complete the missing OASST1 timed block-size/capacity sweeps and save CSV
   results.
3. Train `TrieParrotModel` on `data/oasst1_timed_global_b16`.
4. Compare `model` and `guard` against LRU and Oracle.
5. Add temporal features for model ranking: leaf depth, age, frequency,
   subtree size, last access, matched prefix length, and request length.
6. Explore two-tier KV simulation for GPU/CPU or GPU/SSD cache costs.
