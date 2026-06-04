#!/bin/bash
datasets=("brightkite" "citi" "astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")

mkdir -p logs/benchmark/ppp
max_jobs=4
pids=()
for dataset in "${datasets[@]}"; do
    while (( $(jobs -rp | wc -l) >= max_jobs )); do
        sleep 1
    done
    echo "Running with dataset=$dataset"
    python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred pleco popu pleco-bin --dump_file --output_root_dir stat > "logs/benchmark/ppp/${dataset}.log" 2>&1 &
    pids+=($!)
done

wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name pleco_popu_pleco-bin --fraction 1 --results_dir stat
