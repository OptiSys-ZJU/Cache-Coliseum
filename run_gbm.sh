#!/bin/bash
declare -A gbm_dict

gbm_dict["astar"]="0.001 0.00475 0.01 1"
gbm_dict["bwaves"]="0.01 0.02 0.021 1"
gbm_dict["cactusadm"]="0.001 0.01 0.05 1"
gbm_dict["gems"]="0.1 0.7617835 0.761784 1"
gbm_dict["lbm"]="0.01 0.03 0.1 1"
gbm_dict["leslie3d"]="0.001 0.009 0.01 1"
gbm_dict["mcf"]="0.0001 0.00085 0.001 1"
gbm_dict["omnetpp"]="0.01 0.02 0.05 1"
gbm_dict["sphinx3"]="0.01 0.12 0.14 1"
gbm_dict["xalanc"]="0.01 0.07 0.1 1"

mkdir -p logs/benchmark/gbm
for dataset in "${!gbm_dict[@]}"; do
    fractions=(${gbm_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        echo "Running $pred with dataset=$dataset with fraction $fraction"
        nohup python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred gbm --model_fraction $fraction --dump_file --output_root_dir stat > "logs/benchmark/gbm/${dataset}_${fraction}.log" 2>&1 &
    done
done