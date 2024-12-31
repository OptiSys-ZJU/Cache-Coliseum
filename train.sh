#!/bin/bash
model=$1
datasets=("brightkite" "citi" "astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
fractions=("1" "0.1" "0.01" "0.001")

mkdir -p logs/$model
for dataset in "${datasets[@]}"; do
    for fraction in "${fractions[@]}"; do
        echo "Running $model with dataset=$dataset with fraction $fraction"
        nohup python -m model.$model --dataset "$dataset" --model_fraction $fraction > "logs/$model/${dataset}_${fraction}.log" 2>&1 &
    done
done