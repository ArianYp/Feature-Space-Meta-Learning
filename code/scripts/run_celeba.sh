#!/usr/bin/env bash
# HSFM on CelebA with paper defaults (Table 11).
set -euo pipefail
cd "$(dirname "$0")/.."

export TORCH_HOME="$(pwd)/cache/torch"
export HF_HOME="$(pwd)/cache/huggingface"
export XDG_CACHE_HOME="$(pwd)/cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python train_hsfm.py \
  --dataset celeba \
  --celeba_root_dir "${CELEBA_ROOT:-../DaC/celeba}" \
  --model_checkpoint "${MODEL_CHECKPOINT:-checkpoints/celeba_erm.pt}" \
  --seed "${SEED:-10}" \
  "$@"
