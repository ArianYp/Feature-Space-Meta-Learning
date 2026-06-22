#!/usr/bin/env bash
# HSFM on Dominoes with paper defaults (Table 11).
set -euo pipefail
cd "$(dirname "$0")/.."

export TORCH_HOME="$(pwd)/cache/torch"
export HF_HOME="$(pwd)/cache/huggingface"
export XDG_CACHE_HOME="$(pwd)/cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

python train_hsfm.py \
  --dataset dominoes \
  --dominoes_data_dir "${DOMINOES_ROOT:-Dominoes_SP90 2}" \
  --model_checkpoint "${MODEL_CHECKPOINT:-checkpoints/dominoes_erm.pt}" \
  --seed "${SEED:-60}" \
  "$@"
