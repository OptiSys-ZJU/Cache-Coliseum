#!/usr/bin/env bash
set -euo pipefail

# Full-data OASST1 Trie-PARROT DAgger training entrypoint for the remote A100
# host. Run this from the repository root after syncing code and data.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRAIN_DEVICES="${TRAIN_DEVICES:-single}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_name", torch.cuda.get_device_name(0))
PY

python -m model.trie_model \
  --dataset oasst1_timed_global_b16 \
  --data_root_dir data \
  --device cuda:0 \
  --train_devices "${TRAIN_DEVICES}" \
  --model_config_path configs/full_dagger_oasst1_b16_c256.json \
  --checkpoints_root_dir checkpoints/full_dagger_oasst1_b16_c256
