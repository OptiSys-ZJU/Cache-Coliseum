#!/bin/bash
pred=$1
datasets=("brightkite" "citi" "astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
fractions=("1" "0.1" "0.01" "0.001")

# datasets=("brightkite")
# fractions=("1")

mkdir -p logs/benchmark/$pred
for dataset in "${datasets[@]}"; do
    for fraction in "${fractions[@]}"; do
        echo "Running $pred with dataset=$dataset with fraction $fraction"
        nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred $pred --model_fraction $fraction --dump_file --output_root_dir stat > "logs/benchmark/$pred/${dataset}_${fraction}.log" 2>&1 &
    done
done