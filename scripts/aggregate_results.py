#!/usr/bin/env python
"""Aggregate per-dataset benchmark CSVs into a per-algorithm mean table.

Reads flat CSV files from ``stat/<dataset>_<name>_<fraction>.csv``,
computes per-algorithm mean of Hit Rate, Cost Ratio, and LRU-normalized
Cost Ratio across datasets.

Cross-checks with log files to ensure each dataset actually completed
successfully in the most recent run.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime


CSV_COST_COL = "Cost Ratio"
DISPLAY_COLS = ["Hit Rate", "Cost Ratio", "LRU-norm Cost Ratio"]

SPEC2006_DATASETS = [
    "astar", "bwaves", "bzip", "cactusadm", "gems",
    "lbm", "leslie3d", "libq", "mcf", "milc",
    "omnetpp", "sphinx3", "xalanc",
]


def log_has_result_table(log_path):
    """Check if a log file contains a valid result table (PrettyTable output)."""
    if not os.path.isfile(log_path):
        return False, "log file not found"
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("|") and "Hit Rate" in line:
                return True, None
    return False, "no result table in log"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="CSV basename (real mode: joined predictor names; oracle mode: noise type)")
    parser.add_argument("--fraction", default="1",
                        help="model_fraction subdirectory name (default: 1)")
    parser.add_argument("--results_dir", default="stat",
                        help="root directory passed to benchmark as --output_root_dir")
    parser.add_argument("--logs_dir", default=None,
                        help="directory containing per-dataset log files (default: logs/benchmark/<name>)")
    parser.add_argument("--expected", default="spec2006",
                        choices=["spec2006", "none"],
                        help="expected dataset set for completeness check (default: spec2006)")
    args = parser.parse_args()
    if args.logs_dir is None:
        args.logs_dir = os.path.join("logs", "benchmark", args.name)

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # --- Determine expected datasets ---
    if args.expected == "spec2006":
        expected = SPEC2006_DATASETS
    else:
        expected = sorted(set(
            f.split("_")[0] for f in os.listdir(args.results_dir)
            if f.endswith(f"_{args.name}_{args.fraction}.csv")
        ))

    # --- Verify each dataset has a valid log with result table ---
    failed = []
    for dataset in expected:
        log_path = os.path.join(args.logs_dir, f"{dataset}_{args.fraction}.log")
        ok, reason = log_has_result_table(log_path)
        if not ok:
            failed.append((dataset, reason))

    if failed:
        print("ERROR: The following datasets do not have valid results in their log files:")
        for dataset, reason in failed:
            log_path = os.path.join(args.logs_dir, f"{dataset}_{args.fraction}.log")
            print(f"  {dataset:<15} ({reason}: {log_path})")
        print()
        print("Aggregation aborted. Re-run the benchmark for the failed datasets first.")
        sys.exit(1)

    # --- Collect CSVs ---
    found = []
    mtimes = {}
    for dataset in expected:
        path = os.path.join(args.results_dir, f"{dataset}_{args.name}_{args.fraction}.csv")
        if os.path.isfile(path):
            found.append((dataset, path))
            mtimes[dataset] = os.path.getmtime(path)
        else:
            failed.append((dataset, "CSV not found"))

    if not found:
        print(f"ERROR: No CSVs found matching {args.results_dir}/<dataset>_{args.name}_{args.fraction}.csv")
        sys.exit(1)

    # --- Staleness check ---
    if mtimes:
        times = list(mtimes.values())
        min_t, max_t = min(times), max(times)
        if max_t - min_t > 1800:
            print("WARNING: Result files have timestamps spanning >30 min — possible mix of different runs:")
            for dataset, t in sorted(mtimes.items(), key=lambda x: x[1]):
                print(f"  {dataset:<15} {datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')}")
            print()

    # --- Aggregation with LRU-normalized Cost Ratio ---
    # Per-dataset: collect (alg -> {hit_rate, cost_ratio}) and LRU's cost_ratio
    sums = defaultdict(lambda: [0.0] * len(DISPLAY_COLS))
    counts = defaultdict(int)
    order = []
    errors = []
    rows_per_dataset = {}

    for dataset, path in found:
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                dataset_rows = {}
                dataset_row_count = 0
                for row in reader:
                    name = row["Name"]
                    hit_rate = float(row["Hit Rate"])
                    cost_ratio = float(row[CSV_COST_COL])
                    if math.isnan(hit_rate) or math.isinf(hit_rate):
                        errors.append((dataset, f"non-finite Hit Rate={hit_rate} in row '{name}'"))
                        continue
                    if math.isnan(cost_ratio) or math.isinf(cost_ratio):
                        errors.append((dataset, f"non-finite {CSV_COST_COL}={cost_ratio} in row '{name}'"))
                        continue
                    dataset_rows[name] = (hit_rate, cost_ratio)
                    dataset_row_count += 1
                rows_per_dataset[dataset] = dataset_row_count

                # Compute LRU-normalized Cost Ratio for this dataset
                lru_cost = dataset_rows.get("LRU", (None, None))[1]
                opt_cost = 1.0  # OPT always has cost ratio = 1.0

                for name, (hit_rate, cost_ratio) in dataset_rows.items():
                    if name not in sums:
                        order.append(name)
                    if lru_cost is not None and lru_cost != opt_cost:
                        lru_norm = (cost_ratio - opt_cost) / (lru_cost - opt_cost)
                    else:
                        lru_norm = 0.0
                    sums[name][0] += hit_rate
                    sums[name][1] += cost_ratio
                    sums[name][2] += lru_norm
                    counts[name] += 1

        except (KeyError, ValueError) as e:
            errors.append((dataset, str(e)))
            rows_per_dataset[dataset] = 0

    # --- Empty / corrupt CSV check (hard failure) ---
    empty = [(d, p) for (d, p) in found if rows_per_dataset.get(d, 0) == 0]
    if empty:
        print("ERROR: The following datasets have CSV files with no usable data rows:")
        for d, p in empty:
            try:
                size = os.path.getsize(p)
            except OSError:
                size = -1
            print(f"  {d:<15} (size={size} bytes): {p}")
        print()
        print("Aggregation aborted. Re-run the benchmark for these datasets first.")
        sys.exit(1)

    # --- Row-count consistency check (warning) ---
    if rows_per_dataset:
        all_counts = set(rows_per_dataset.values())
        if len(all_counts) > 1:
            max_n = max(all_counts)
            print("WARNING: Datasets have different numbers of algorithm rows — "
                  "some runs may have partially failed:")
            for d, n in sorted(rows_per_dataset.items()):
                marker = "" if n == max_n else "  <-- INCOMPLETE"
                print(f"  {d:<15} {n} rows{marker}")
            print()

    if errors:
        print("WARNING: Failed to parse some CSV rows:")
        for dataset, err in errors:
            print(f"  {dataset}: {err}")
        print()

    if not order:
        print("ERROR: No valid data after parsing.")
        sys.exit(1)

    print(f"Aggregated {len(found)} datasets for name='{args.name}', fraction='{args.fraction}':")
    print("  " + ", ".join(d for d, _ in found))
    print()

    header = ["Name", "N"] + DISPLAY_COLS
    col_widths = [max(len(h), 40 if h == "Name" else 10) for h in header]

    def fmt_row(vals):
        return "| " + " | ".join(
            v.ljust(w) if isinstance(v, str) else v.rjust(w)
            for v, w in zip(vals, col_widths)
        ) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(sep)
    print(fmt_row(header))
    print(sep)
    for name in order:
        n = counts[name]
        row = [name, str(n)] + [f"{sums[name][i] / n:.4f}" for i in range(len(DISPLAY_COLS))]
        print(fmt_row(row))
    print(sep)


if __name__ == "__main__":
    main()
