import argparse
import csv
from pathlib import Path


INPUT_SPECS = [
    (
        "lru",
        "baseline",
        Path("res/oasst1_timed_global_b16_c256_full_valid_baselines_verify.csv"),
        "lru",
    ),
    (
        "oracle",
        "baseline",
        Path("res/oasst1_timed_global_b16_c256_full_valid_baselines_verify.csv"),
        "oracle",
    ),
    (
        "model_best",
        "best.ckpt",
        Path("res/oasst1_timed_global_b16_c256_best_t02_verify.csv"),
        "model",
    ),
    (
        "guard_best_t0.2",
        "best.ckpt",
        Path("res/oasst1_timed_global_b16_c256_best_t02_verify.csv"),
        "guard",
    ),
    (
        "model_step225",
        "step_225.ckpt",
        Path("res/oasst1_timed_global_b16_c256_step225_t005_verify.csv"),
        "model",
    ),
    (
        "guard_step225_t0.05",
        "step_225.ckpt",
        Path("res/oasst1_timed_global_b16_c256_step225_t005_verify.csv"),
        "guard",
    ),
]


def load_row(path: Path, policy: str):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("policy") == policy:
                return row
    raise ValueError(f"policy={policy} not found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize full-valid b16/c256 model/guard results into one CSV."
    )
    parser.add_argument(
        "--output_csv",
        default="res/oasst1_timed_global_b16_c256_full_valid_comparison.csv",
    )
    args = parser.parse_args()

    rows = []
    lru_hit_rate = None
    oracle_hit_rate = None

    loaded = []
    for label, checkpoint, path, policy in INPUT_SPECS:
        row = load_row(path, policy)
        loaded.append((label, checkpoint, row))
        if label == "lru":
            lru_hit_rate = float(row["block_hit_rate"])
        elif label == "oracle":
            oracle_hit_rate = float(row["block_hit_rate"])

    if lru_hit_rate is None or oracle_hit_rate is None:
        raise ValueError("Both LRU and oracle rows are required")

    oracle_gap = oracle_hit_rate - lru_hit_rate
    for label, checkpoint, row in loaded:
        hit_rate = float(row["block_hit_rate"])
        gain_vs_lru = hit_rate - lru_hit_rate
        recovered_gap = (
            gain_vs_lru / oracle_gap if oracle_gap > 0 else 0.0
        )
        rows.append({
            "label": label,
            "checkpoint": checkpoint,
            "policy": row["policy"],
            "dataset": "oasst1_timed_global_b16",
            "split": "valid",
            "capacity": row["capacity"],
            "capacity_tokens": row["capacity_tokens"],
            "block_token_size": row["block_token_size"],
            "block_hit_rate": row["block_hit_rate"],
            "request_full_hit_rate": row["request_full_hit_rate"],
            "avg_prefix_hit_len": row["avg_prefix_hit_len"],
            "guard_rate": row.get("guard_rate", "0.0"),
            "gain_vs_lru": gain_vs_lru,
            "oracle_minus_lru": oracle_gap,
            "recovered_oracle_gap_fraction": recovered_gap,
        })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
