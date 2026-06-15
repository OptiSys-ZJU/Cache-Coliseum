import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_list(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def read_guard_row(path: Path):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("policy") == "guard":
                return row
    raise ValueError(f"guard row not found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep guard thresholds/checkpoints for OASST1 b16 c256 full-valid."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default="oasst1_timed_global_b16")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated checkpoint filenames under checkpoints/trie_model/<dataset>/",
    )
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Comma-separated float thresholds",
    )
    parser.add_argument(
        "--model_config_path",
        default="checkpoints/trie_model/oasst1_timed_global_b16/config.json",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="checkpoints/trie_model/oasst1_timed_global_b16",
    )
    parser.add_argument(
        "--output_csv",
        default="res/oasst1_timed_global_b16_c256_guard_fine_sweep.csv",
    )
    parser.add_argument(
        "--tmp_dir",
        default="res/tmp_oasst1_b16_guard_sweep",
    )
    args = parser.parse_args()

    checkpoints = parse_list(args.checkpoints)
    thresholds = [float(item) for item in parse_list(args.thresholds)]

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint_name in checkpoints:
        checkpoint_path = Path(args.checkpoints_dir) / checkpoint_name
        for threshold in thresholds:
            tmp_csv = tmp_dir / f"{checkpoint_name}_t{str(threshold).replace('.', '_')}.csv"
            command = [
                args.python,
                "-m",
                "benchmark.trie_kv",
                "--dataset",
                args.dataset,
                "--split",
                args.split,
                "--capacity",
                str(args.capacity),
                "--policy",
                "guard",
                "--model_config_path",
                args.model_config_path,
                "--model_checkpoint_path",
                str(checkpoint_path),
                "--variance_threshold",
                str(threshold),
                "--output_csv",
                str(tmp_csv),
            ]
            print("run", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            row = read_guard_row(tmp_csv)
            rows.append({
                "checkpoint": checkpoint_name,
                "variance_threshold": threshold,
                "block_hit_rate": row["block_hit_rate"],
                "request_full_hit_rate": row["request_full_hit_rate"],
                "avg_prefix_hit_len": row["avg_prefix_hit_len"],
                "guard_rate": row["guard_rate"],
                "recompute_blocks": row["recompute_blocks"],
                "evictions": row["evictions"],
            })

    rows.sort(
        key=lambda item: (
            -float(item["block_hit_rate"]),
            item["checkpoint"],
            float(item["variance_threshold"]),
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
