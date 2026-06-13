# OASST1 Timed KV Cache Experiment Manifest

Last synced: 2026-06-13

This document is the handoff point for the current `trie-cache` experiment line.
It records what the generated files mean, which assumptions are currently in
scope, and what the next agent should do without re-reading the whole chat.

## Current Research Line

The original repository studied CPU cache replacement algorithms. The current
project direction is to test whether a trie/prefix-aware structure can improve
eviction decisions for LLM KV-cache style workloads.

The active experiment line is the OASST1 timed request stream. Dense block
identities are scoped globally for the shared-cache simulation. The current
main workload is:

- shared cache capacity: one global KV memory pool shared by all requests
- request order: timestamp order derived from OASST1 conversation events
- identity scope: `global`
- event role: `all`
- policies: `lru`, `rand`, `oracle`
- no round-robin stress interleaving for the main claim
- no legacy `origin-trie-cache` reproduction on the critical path

The defensible claim boundary is:

> Under the OASST1 timestamped shared-cache simulation, there is a sizable
> LRU-to-oracle gap in the middle KV pressure regime. The gap is much smaller
> when cache capacity is extremely small or large. The next task is to train a
> model/guard policy and test whether it can recover part of that oracle gap
> under the same workload and capacity settings.

## Local Branch And Dirty State

Current branch:

```powershell
git branch --show-current
# trie-cache
```

Known local code changes from this experiment line:

- `.gitignore`
  - includes the local Python virtual environment `lkcp/`
- `benchmark/trie_kv.py`
  - improved CSV output directory handling
  - supports timestamp-shared labeling from dataset metadata
  - supports trace interleaving for stress tests, but this is not the main line
- `scripts/data_process/preprocess_oasst1_timed.py`
  - new OASST1 timed preprocessing script
  - aligns OASST1 event/order data with tokenized CSV rows
  - supports block token size, identity scope, and event role

There are also untracked report files under `reports/` and an untracked
`CLAUDE.md`. Do not revert or delete them unless the user explicitly asks.

## Python Environment

Use the existing virtual environment:

```powershell
.\lkcp\Scripts\python.exe
```

Packages installed during this line include at least:

- `datasets`
- `numpy`
- `tqdm`
- `matplotlib`

Before training, confirm whether `torch` is installed and whether CPU or CUDA is
available:

```powershell
.\lkcp\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data Sources

Raw/source files are under `data/traces/`:

- `oasst1_trees.json`
  - original OASST conversation trees
- `oasst1_dialogs.json`
  - root-to-leaf linear dialogues
- `oasst1_sequence.json`
  - dialogue turns flattened into timestamp-sorted access events
- `oasst1_reqs_DeepSeek-R1-Distill-Qwen-14B_unique.json`
  - OpenAI chat request style records
- `oass1_train.csv`
  - tokenized train requests, one CSV row per request
- `oass1_val.csv`
  - tokenized validation requests

Important alignment note:

- validation CSV has one extra leading row compared with the event/request
  sequence; the preprocessing script detects this and uses a row offset.

## Preprocessed OASST1 Timed Datasets

Generated datasets are under directories like:

- `data/oasst1_timed_global_b1`
- `data/oasst1_timed_global_b8`
- `data/oasst1_timed_global_b16`
- `data/oasst1_timed_global_b32`

Directory name convention:

- `oasst1_timed`: timestamped OASST1 request stream
- `global`: global block identity setting
- `bN`: `block_token_size=N`

Expected files in each generated dataset:

- `train.pkl`
  - training split if generated
- `valid.pkl`
  - validation split used for current benchmark runs
- `metadata.json`
  - preprocessing settings and counts
- optional JSONL event dumps
  - debugging and human inspection

Recreate a validation-only dataset:

```powershell
.\lkcp\Scripts\python.exe scripts\data_process\preprocess_oasst1_timed.py `
  --output_dir data\oasst1_timed_global_b16 `
  --source_splits validation `
  --block_token_size 16 `
  --identity_scope global `
  --event_role all
```

For training, generate both train and validation splits for the chosen block
size:

```powershell
.\lkcp\Scripts\python.exe scripts\data_process\preprocess_oasst1_timed.py `
  --output_dir data\oasst1_timed_global_b16 `
  --source_splits train validation `
  --block_token_size 16 `
  --identity_scope global `
  --event_role all
```

## Block And Capacity Semantics

`block_token_size` is the number of token ids grouped into one logical cache
block before constructing the trie sequence.

`capacity_blocks` is the trie cache capacity passed to the benchmark, measured
as cache blocks/nodes in the current simulation. Equal-token-budget comparisons
should use:

```text
capacity_tokens = block_token_size * capacity_blocks
```

This is related to, but not identical to, the original CPU cacheline concept.
For LLM KV-cache interpretation, `block_token_size` is closer to the KV block
granularity used by prefix-cache systems.

Hit rate in the current result CSV is measured over cache blocks:

```text
hit_rate = hit_blocks / total_requested_blocks
```

Recompute block counts in the summary are:

```text
recompute_blocks = total_requested_blocks - hit_blocks
oracle_saved_blocks_vs_lru = lru_recompute_blocks - oracle_recompute_blocks
```

## Benchmark Command

Run LRU/RAND/oracle for one preprocessed dataset:

```powershell
.\lkcp\Scripts\python.exe -m benchmark.trie_kv `
  --dataset oasst1_timed_global_b16 `
  --split valid `
  --capacity 256 512 1024 2048 4096 `
  --policy lru rand oracle `
  --output_csv res\oasst1_timed_global_b16_valid_kv.csv
```

Main summary CSV:

- `res/oasst1_timed_global_blocksize_gap_summary_v2.csv`

Older summary:

- `res/oasst1_timed_global_blocksize_gap_summary.csv`
  - was kept because Windows had locked it during one overwrite attempt

Figures:

- `res/figures/oasst1_hit_rate_by_blocksize.png`
- `res/figures/oasst1_oracle_lru_gap_by_token_budget.png`
- `res/figures/oasst1_gap_equal_token_budget_bars.png`
- `res/figures/oasst1_oracle_saved_tokens.png`
- `res/figures/oasst1_gap_bubble_heatmap.png`

## Current Result Snapshot

Main settings already summarized in
`res/oasst1_timed_global_blocksize_gap_summary_v2.csv`:

| block_token_size | capacity_blocks covered | notes |
|---:|---|---|
| 1 | 256, 512, 1024, 4096 | larger capacities previously timed out |
| 8 | 256, 512, 1024, 2048, 4096, 8192 | broadest current sweep |
| 16 | 256, 512, 1024, 2048, 4096 | useful training candidate |
| 32 | 128, 256, 512, 1024, 2048 | equal-token-budget coverage |

Representative high-gap points:

| block_token_size | capacity_blocks | capacity_tokens | LRU | Oracle | Oracle-LRU |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 4096 | 0.3033 | 0.4750 | 0.1716 |
| 8 | 512 | 4096 | 0.2833 | 0.4567 | 0.1734 |
| 16 | 256 | 4096 | 0.2701 | 0.4428 | 0.1726 |
| 32 | 128 | 4096 | 0.2419 | 0.4136 | 0.1717 |
| 8 | 256 | 2048 | 0.1989 | 0.3438 | 0.1450 |
| 16 | 512 | 8192 | 0.3778 | 0.5061 | 0.1283 |

Current observation:

- The oracle gap is not monotonic in token budget.
- The gap is small when the cache is too tiny because almost everything is
  evicted/recomputed.
- The gap is small when the cache is too large because LRU already keeps most
  useful prefixes.
- The best target region so far is around `capacity_tokens ~= 4096`, with
  block sizes 8, 16, and 32 all giving similar oracle gaps at equal token
  budget.

## Missing Or Incomplete Settings

The user asked to fill missing block size and capacity combinations, including
ones that previously did not finish because of timeout.

Suggested completion matrix:

| block_token_size | capacity_blocks to verify/complete | reason |
|---:|---|---|
| 1 | 2048, 8192, 16384 | fill token-budget curve; large cases were slow |
| 8 | 128, 16384 | optional endpoints around current sweep |
| 16 | 128, 8192 | endpoint and large-token-budget comparison |
| 32 | 4096, 8192 | large-token-budget comparison |

Do not delete existing result CSVs. Write new runs to new files or use a runner
that appends/merges deterministically.

## Training Entry Point

Trie model training entry point:

```powershell
# Create or choose this config before starting training if it does not exist.
.\lkcp\Scripts\python.exe -m model.trie_model `
  --dataset oasst1_timed_global_b16 `
  --data_root_dir data `
  --device cpu `
  --model_config_path checkpoints\trie_model\model_config.json `
  --checkpoints_root_dir checkpoints
```

Known checkpoint behavior from `model/trie_model/__main__.py`:

- checkpoint directory:
  - `checkpoints/trie_model/<dataset>/`
- config copy:
  - `checkpoints/trie_model/<dataset>/config.json`
- best checkpoint:
  - `checkpoints/trie_model/<dataset>/best.ckpt`
- periodic checkpoints:
  - `checkpoints/trie_model/<dataset>/step_<step>.ckpt`
- final checkpoint:
  - `checkpoints/trie_model/<dataset>/final_<step>.ckpt`

Before long training:

1. Confirm `torch` import works.
2. Confirm selected dataset has both `train.pkl` and `valid.pkl`.
3. Run a small smoke training config or low `max_examples` setting if available.
4. Then start the full run and let checkpoints accumulate.

Recommended first training target:

- `data/oasst1_timed_global_b16`
- evaluate at `capacity_blocks=256`
- this corresponds to `capacity_tokens=4096`, where Oracle-LRU gap is large

Alternative target:

- `data/oasst1_timed_global_b8`
- evaluate at `capacity_blocks=512`
- also `capacity_tokens=4096`

## Model Evaluation After Training

After a checkpoint exists, compare `model` and `guard` against LRU/oracle under
the same condition:

```powershell
.\lkcp\Scripts\python.exe -m benchmark.trie_kv `
  --dataset oasst1_timed_global_b16 `
  --split valid `
  --capacity 256 `
  --policy lru oracle model guard `
  --model_config_path checkpoints\trie_model\oasst1_timed_global_b16\config.json `
  --model_checkpoint_path checkpoints\trie_model\oasst1_timed_global_b16\best.ckpt `
  --output_csv res\oasst1_timed_global_b16_valid_model_gap.csv
```

Primary question:

```text
How much of (oracle_hit_rate - lru_hit_rate) does model/guard recover?
```

Useful derived metric:

```text
gap_recovery = (model_hit_rate - lru_hit_rate) / (oracle_hit_rate - lru_hit_rate)
```

## Review Risks To Keep In Mind

- `identity_scope=global` approximates shared prefix cache behavior. It should
  not be described as cross-trace reuse of trace-local ids.
- `event_role=all` means both user and assistant turns contribute cache events.
  This approximates cache state evolution, not necessarily external user-facing
  request count.
- Current hit rate is block-level, not request-level.
- Equal `capacity_blocks` across different block sizes is not equal token
  budget. Use `capacity_tokens` for cross-block-size comparison.
- Round-robin interleaving creates an exaggerated LRU-bad stress test. Keep it
  separate from the main OASST timed claim.
- If training produces poor results, first audit whether model inference uses
  the same cache state/features as training and whether `TrieModelPredictAlgorithm`
  updates history with enough context.

## Next Handoff Checklist

1. Fill missing settings and regenerate/merge the summary CSV.
2. Add or use a deterministic experiment runner so interrupted sweeps can resume.
3. Confirm PyTorch availability.
4. Generate train+valid preprocessed data for the selected block size.
5. Start checkpointed training.
6. Evaluate `model` and `guard` at high-gap token budgets, especially:
   - `b16, capacity_blocks=256`
   - `b8, capacity_blocks=512`
   - optionally `b32, capacity_blocks=128`
7. Iterate on model/training only after verifying LRU/oracle/model are evaluated
   on exactly the same dataset, split, capacity, and block size.
