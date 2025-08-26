#!/usr/bin/env bash
set -euo pipefail

# knobs (can override via env: DATASET=..., OUT_DIR=..., etc.)
DATASET="${DATASET:-CSL_News}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-16}"
GAS="${GAS:-8}"
LR="${LR:-3e-4}"
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
OUT_DIR="${OUT_DIR:-$PWD/out/stage1_pretraining_advanced}"
NUM_WORKERS="${NUM_WORKERS:-12}"   # per GPU → 4*12 = 48 workers total
PIN_MEM="${PIN_MEM:-1}"

cd "$(dirname "$0")/.."
[ -e dataset ] || ln -s data dataset

# Keep shells from fighting each other
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"

# Guardrails: no user-site, no stray PYTHONPATH, tame threads
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Offline friendly (ok if you don’t need HF network)
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Optional: pin 4 GPUs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ulimit -n 131072 || true

echo "Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Writing to $OUT_DIR"

# IMPORTANT: use `python -s -m deepspeed` (the `-s` kills user-site)
exec python -s -u -m deepspeed --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" pre_training.py \
  --batch-size "$BATCH" \
  --gradient-accumulation-steps "$GAS" \
  --epochs "$EPOCHS" \
  --opt AdamW \
  --lr "$LR" \
  --sched "$SCHED" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --output_dir "$OUT_DIR" \
  --dataset "$DATASET" \
  ${PIN_MEM:+--pin-mem}

