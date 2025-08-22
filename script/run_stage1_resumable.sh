#!/usr/bin/env bash
set -euo pipefail

# --- SETTINGS YOU MAY TWEAK ---
TOTAL_EPOCHS="${1:-20}"
OUT_DIR="${2:-$PWD/out/stage1_pretraining}"
DATASET="${3:-CSL_News}"
BATCH=16
GAS=8
LR=3e-4
NUM_WORKERS=9          # per process; with 4 GPUs -> ~36 workers total
# -------------------------------

cd "$(dirname "$0")/.."   # cd to repo root

# Ensure expected path exists (code reads ./dataset)
[ -e dataset ] || ln -s data dataset

# Find the latest completed epoch
LAST=0
if [ -d "$OUT_DIR" ]; then
  if [ -f "$OUT_DIR/LAST_EPOCH" ]; then
    LAST="$(cat "$OUT_DIR/LAST_EPOCH" || echo 0)"
  else
    LAST="$(ls -1 "$OUT_DIR" 2>/dev/null | grep -E '^epoch_[0-9]{3}$' | sed 's/epoch_//' | sort -n | tail -1 || echo 0)"
  fi
fi
LAST="${LAST:-0}"
REMAIN=$(( TOTAL_EPOCHS - LAST ))
if (( REMAIN <= 0 )); then
  echo "Nothing to do: already at/beyond ${TOTAL_EPOCHS} epochs (LAST_EPOCH=$LAST)."
  exit 0
fi

FINETUNE_ARG=()
if (( LAST > 0 )); then
  CKPT_DIR="${OUT_DIR}/epoch_$(printf '%03d' "$LAST")"
  if [ -d "$CKPT_DIR" ]; then
    FINETUNE_ARG=( --finetune "$CKPT_DIR" )
    echo "Resuming from epoch $LAST via --finetune $CKPT_DIR; training $REMAIN more epoch(s)."
  else
    echo "NOTE: $CKPT_DIR not found; starting fresh."
  fi
else
  echo "Starting fresh for $TOTAL_EPOCHS epoch(s)."
fi

# Prefer a random MASTER_PORT to avoid cross-tab collisions
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( ( RANDOM % 45000 ) + 20000 ))}"

# Optional GPU pinning:
# export CUDA_VISIBLE_DEVICES=0,1,2,3

exec deepspeed pre_training.py \
  --batch-size "$BATCH" \
  --gradient-accumulation-steps "$GAS" \
  --epochs "$REMAIN" \
  --opt AdamW \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --output_dir "$OUT_DIR" \
  --dataset "$DATASET" \
  "${FINETUNE_ARG[@]}"

