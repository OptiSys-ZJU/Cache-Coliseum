import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MATRIX = {
    1: [2048, 8192, 16384],
    8: [128, 16384],
    16: [128, 8192],
    32: [4096, 8192],
}


def parse_matrix(items):
    if not items:
        return DEFAULT_MATRIX

    matrix = {}
    for item in items:
        block, capacities = item.split(":", 1)
        matrix[int(block)] = [
            int(capacity)
            for capacity in capacities.split(",")
            if capacity.strip()
        ]
    return matrix


def existing_policies(path):
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row.get("policy") for row in csv.DictReader(f)}


def complete(path):
    return {"lru", "rand", "oracle"}.issubset(existing_policies(path))


def main():
    parser = argparse.ArgumentParser(
        description="Resume missing OASST1 timed lru/rand/oracle capacity runs."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--res_dir", default="res")
    parser.add_argument("--matrix", nargs="*", help="Items like 16:128,8192")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    matrix = parse_matrix(args.matrix)

    for block_token_size, capacities in sorted(matrix.items()):
        dataset = f"oasst1_timed_global_b{block_token_size}"
        for capacity in capacities:
            output_csv = (
                res_dir /
                f"{dataset}_{args.split}_kv_missing_{capacity}.csv"
            )
            if not args.force and complete(output_csv):
                print(f"skip complete {output_csv}")
                continue

            command = [
                args.python,
                "-m",
                "benchmark.trie_kv",
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--capacity",
                str(capacity),
                "--policy",
                "lru",
                "rand",
                "oracle",
                "--output_csv",
                str(output_csv),
            ]
            print("run", " ".join(command), flush=True)
            start = time.time()
            subprocess.run(command, check=True)
            elapsed = time.time() - start
            print(f"done {output_csv} elapsed_seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
