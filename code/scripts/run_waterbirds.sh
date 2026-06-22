#!/usr/bin/env bash
# HSFM on Waterbirds with paper defaults (Table 11).
set -euo pipefail
cd "$(dirname "$0")/.."

export TORCH_HOME="$(pwd)/cache/torch"
export HF_HOME="$(pwd)/cache/huggingface"
export XDG_CACHE_HOME="$(pwd)/cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python train_hsfm.py \
  --dataset waterbirds \
  --waterbirds_root_dir "${WATERBIRDS_ROOT:-../DaC/waterbird_complete95_forest2water2}" \
  --model_checkpoint "${MODEL_CHECKPOINT:-checkpoints/waterbirds_erm.pt}" \
  --seed "${SEED:-60}" \
  "$@"
