#!/bin/bash
datasets=("brightkite" "citi" "astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")

mkdir -p logs/benchmark/ppp
for dataset in "${datasets[@]}"; do
    echo "Running with dataset=$dataset"
    nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred pleco popu pleco-bin --test_all --dump_file --output_root_dir stat > "logs/benchmark/ppp/${dataset}.log" 2>&1 &
done