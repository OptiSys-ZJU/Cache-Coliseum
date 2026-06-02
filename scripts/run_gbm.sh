#!/bin/bash
# Usage: scripts/run_gbm.sh [MODEL_FRACTION]
# MODEL_FRACTION defaults to 1 (full training set).
fraction="${1:-1}"

datasets=(astar bwaves bzip cactusadm gems lbm leslie3d libq mcf milc omnetpp sphinx3 xalanc)

mkdir -p logs/benchmark/gbm
pids=()
for dataset in "${datasets[@]}"; do
    echo "Running with dataset=$dataset with fraction $fraction"
    nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred gbm --model_fraction "$fraction" --dump_file --output_root_dir stat > "logs/benchmark/gbm/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name gbm --fraction "$fraction" --results_dir stat
