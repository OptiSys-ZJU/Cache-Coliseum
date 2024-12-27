#!/bin/bash
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc" "brightkite" "citi")
noise_types=("dis" "bin" "logdis")

for dataset in "${datasets[@]}"; do
  for noise_type in "${noise_types[@]}"; do
    echo "Running with dataset=$dataset and noise_type=$noise_type"
    nohup python -m benchmark --dataset "$dataset" --oracle --noise_type "$noise_type" > "${dataset}_${noise_type}.log" 2>&1 &
  done
  echo "Running with dataset=$dataset and real"
  nohup python -m benchmark --dataset "$dataset" --real > "${dataset}_real.log" 2>&1 &
done