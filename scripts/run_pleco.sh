#!/bin/bash
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")

mkdir -p logs/benchmark/pleco
pids=()
for dataset in "${datasets[@]}"; do
    echo "Running with dataset=$dataset"
    nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred pleco --dump_file --output_root_dir stat > "logs/benchmark/pleco/${dataset}.log" 2>&1 &
    pids+=($!)
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name pleco --fraction 1 --results_dir stat
