#!/bin/bash
# Train TrieParrotModel on YooChoose dataset
# Usage: bash scripts/train_trie_model.sh [device]
#   device: cpu (default), cuda:0, cuda:1, etc.

set -e

DEVICE=${1:-"cpu"}
DATASET="yoochoose"
CONFIG_PATH="checkpoints/trie_model/model_config.json"
CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"

mkdir -p logs/trie_model

echo "=============================="
echo "Training TrieParrotModel"  
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Config: $CONFIG_PATH"
echo "=============================="

python -m model.trie_model \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --model_config_path "$CONFIG_PATH" \
    --checkpoints_root_dir "$CHECKPOINT_DIR" \
    --data_root_dir "$DATA_DIR" \
    2>&1 | tee "logs/trie_model/${DATASET}_train.log"

echo "Training complete. Logs saved to logs/trie_model/${DATASET}_train.log"
