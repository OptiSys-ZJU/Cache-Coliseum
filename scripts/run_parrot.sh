#!/bin/bash
# Usage: scripts/run_parrot.sh [MODEL_FRACTION]
# MODEL_FRACTION defaults to 1 (full training set).
fraction="${1:-1}"

datasets=(astar bwaves bzip cactusadm gems lbm leslie3d libq mcf milc omnetpp sphinx3 xalanc)

mkdir -p logs/benchmark/parrot

cuda_devices=("cuda:0" "cuda:1")
cuda_index=0

# Concurrency cap. Parrot runs on GPU, so default to one job per device
# (= number of cuda_devices) to avoid OOM; raise if your GPUs have headroom.
max_jobs=${#cuda_devices[@]}
pids=()
for dataset in "${datasets[@]}"; do
    while (( $(jobs -rp | wc -l) >= max_jobs )); do
        sleep 1
    done
    current_device=${cuda_devices[$cuda_index]}
    echo "Running parrot with dataset=$dataset with fraction $fraction on device $current_device"
    python -m benchmark --boost --boost_fr --dataset "$dataset" --device "$current_device" --real --pred parrot --model_fraction "$fraction" --dump_file --output_root_dir stat > "logs/benchmark/parrot/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
    ((cuda_index=(cuda_index+1)%2))
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name parrot --fraction "$fraction" --results_dir stat
