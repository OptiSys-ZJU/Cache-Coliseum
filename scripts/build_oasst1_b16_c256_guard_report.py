import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_policy_row(path: Path, policy: str):
    for row in read_rows(path):
        if row.get("policy") == policy:
            return row
    raise ValueError(f"policy={policy} not found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a final b16/c256 full-valid comparison report."
    )
    parser.add_argument(
        "--output_csv",
        default="res/oasst1_timed_global_b16_c256_full_valid_guard_report.csv",
    )
    args = parser.parse_args()

    lru = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_full_valid_baselines_verify.csv"),
        "lru",
    )
    oracle = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_full_valid_baselines_verify.csv"),
        "oracle",
    )
    step225_model = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_step225_t005_verify.csv"),
        "model",
    )
    best_model = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_best_t02_verify.csv"),
        "model",
    )
    step225_guard = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_step225_t005_verify.csv"),
        "guard",
    )
    best02_guard = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_best_t02_verify.csv"),
        "guard",
    )
    best015_guard = find_policy_row(
        Path("res/oasst1_timed_global_b16_c256_best_t015_verify.csv"),
        "guard",
    )

    lru_hit = float(lru["block_hit_rate"])
    oracle_hit = float(oracle["block_hit_rate"])
    oracle_gap = oracle_hit - lru_hit

    entries = [
        ("lru", "baseline", None, lru),
        ("oracle", "baseline", None, oracle),
        ("model_best", "best.ckpt", None, best_model),
        ("model_step225", "step_225.ckpt", None, step225_model),
        ("guard_prev_best", "step_225.ckpt", 0.05, step225_guard),
        ("guard_prev_alt", "best.ckpt", 0.2, best02_guard),
        ("guard_new_best", "best.ckpt", 0.15, best015_guard),
    ]

    rows = []
    for label, checkpoint, threshold, row in entries:
        hit = float(row["block_hit_rate"])
        gain = hit - lru_hit
        rows.append({
            "label": label,
            "checkpoint": checkpoint,
            "variance_threshold": "" if threshold is None else threshold,
            "policy": row["policy"],
            "block_hit_rate": row["block_hit_rate"],
            "request_full_hit_rate": row["request_full_hit_rate"],
            "avg_prefix_hit_len": row["avg_prefix_hit_len"],
            "guard_rate": row.get("guard_rate", "0.0"),
            "gain_vs_lru": gain,
            "oracle_minus_lru": oracle_gap,
            "recovered_oracle_gap_fraction": gain / oracle_gap if oracle_gap else 0.0,
        })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
