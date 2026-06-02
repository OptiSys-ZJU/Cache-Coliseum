#!/bin/bash
# Usage: scripts/run_lrb.sh [MODEL_FRACTION]
# MODEL_FRACTION defaults to 1 (full training set).
fraction="${1:-1}"

datasets=(astar bwaves bzip cactusadm gems lbm leslie3d libq mcf milc omnetpp sphinx3 xalanc)

mkdir -p logs/benchmark/lrb
pids=()
for dataset in "${datasets[@]}"; do
    echo "Running with dataset=$dataset with fraction $fraction"
    nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred lrb --model_fraction "$fraction" --dump_file --output_root_dir stat > "logs/benchmark/lrb/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name lrb --fraction "$fraction" --results_dir stat
