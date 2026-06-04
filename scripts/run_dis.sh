#!/bin/bash
# datasets=("brightkite" "citi")
datasets=("brightkite" "citi" "astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")

mkdir -p logs/benchmark/oracle
max_jobs=4
pids=()
for dataset in "${datasets[@]}"; do
    while (( $(jobs -rp | wc -l) >= max_jobs )); do
        sleep 1
    done
    echo "Running with dataset=$dataset"
    python -m benchmark --boost_fr --dataset "$dataset" --oracle --pred oracle_dis --noise_type dis --dump_file --output_root_dir stat > "logs/benchmark/oracle/${dataset}_dis.log" 2>&1 &
    pids+=($!)
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name dis --fraction 1 --results_dir stat
