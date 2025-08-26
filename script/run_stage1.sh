#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Uni-Sign expects ./dataset; keep the symlink fresh
[ -e dataset ] || ln -s data dataset

# GPUs/CPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}

# Absolute output dir to avoid surprises if run from another cwd
OUT_DIR="${OUT_DIR:-$PWD/out/stage1_pretraining}"

# Launch DeepSpeed (CLI)
deepspeed --num_gpus "$NUM_GPUS" pre_training.py \
  --batch-size 16 \
  --gradient-accumulation-steps 8 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 2048 \
  --num_workers "$NUM_WORKERS" \
  --output_dir "$OUT_DIR" \
  --dataset CSL_News \
  \
  "$@"
