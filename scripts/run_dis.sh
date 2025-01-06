#!/bin/bash
datasets=("brightkite" "citi")

mkdir -p logs/benchmark/oracle
for dataset in "${datasets[@]}"; do
    echo "Running with dataset=$dataset"
    nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --oracle --pred oracle_dis --noise_type dis --test_all --dump_file --output_root_dir stat > "logs/benchmark/oracle/${dataset}_dis.log" 2>&1 &
done