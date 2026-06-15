import argparse
import csv
import os
import re
from pathlib import Path


DATASET_RE = re.compile(r"oasst1_timed_global_b(?P<block>\d+)")
POLICIES = ("lru", "rand", "oracle")


def read_rows(res_dir: Path):
    rows_by_key = {}
    for path in sorted(res_dir.glob("oasst1_timed_global_b*_valid_kv*.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("dataset", "")
                match = DATASET_RE.search(dataset)
                if not match:
                    continue
                key = (
                    dataset,
                    row.get("split", ""),
                    row.get("trace_mode", ""),
                    row.get("policy", ""),
                    int(row.get("capacity", 0)),
                )
                rows_by_key[key] = {
                    **row,
                    "block_token_size": int(match.group("block")),
                    "source_csv": path.name,
                    "source_mtime": path.stat().st_mtime,
                }
    return rows_by_key.values()


def make_summary(rows):
    grouped = {}
    for row in rows:
        if row.get("split") != "valid":
            continue
        if row.get("trace_mode") != "timestamp_shared":
            continue
        policy = row.get("policy")
        if policy not in POLICIES:
            continue
        key = (row["block_token_size"], int(row["capacity"]))
        grouped.setdefault(key, {})[policy] = row

    summary = []
    for (block_token_size, capacity), by_policy in grouped.items():
        if not all(policy in by_policy for policy in POLICIES):
            continue
        lru = by_policy["lru"]
        oracle = by_policy["oracle"]
        lru_hit = float(lru["block_hit_rate"])
        rand_hit = float(by_policy["rand"]["block_hit_rate"])
        oracle_hit = float(oracle["block_hit_rate"])
        lru_recompute = int(float(lru["recompute_blocks"]))
        oracle_recompute = int(float(oracle["recompute_blocks"]))
        gap = oracle_hit - lru_hit
        relative_gain = (gap / lru_hit * 100.0) if lru_hit else 0.0
        summary.append({
            "block_token_size": block_token_size,
            "capacity_blocks": capacity,
            "capacity_tokens": block_token_size * capacity,
            "lru": lru_hit,
            "rand": rand_hit,
            "oracle": oracle_hit,
            "oracle_minus_lru": gap,
            "oracle_relative_gain_pct": relative_gain,
            "lru_recompute_blocks": lru_recompute,
            "oracle_recompute_blocks": oracle_recompute,
            "oracle_saved_blocks_vs_lru": lru_recompute - oracle_recompute,
        })
    return sorted(summary, key=lambda r: (r["block_token_size"], r["capacity_blocks"]))


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Merge OASST1 timed valid KV result CSVs into a gap summary."
    )
    parser.add_argument("--res_dir", default="res")
    parser.add_argument(
        "--output_csv",
        default=os.path.join("res", "oasst1_timed_global_blocksize_gap_summary_v3.csv"),
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.res_dir))
    summary = make_summary(rows)
    write_csv(Path(args.output_csv), summary)

    covered = {}
    for row in summary:
        covered.setdefault(row["block_token_size"], []).append(row["capacity_blocks"])

    print(f"wrote {args.output_csv} rows={len(summary)}")
    for block_token_size in sorted(covered):
        capacities = ",".join(str(item) for item in sorted(covered[block_token_size]))
        print(f"b{block_token_size}: {capacities}")


if __name__ == "__main__":
    main()
