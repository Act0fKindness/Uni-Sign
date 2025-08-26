#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   tools/uni_sign_train.sh stage2        # RGB-pose pretraining on CSL_News
#   tools/uni_sign_train.sh stage3        # fine-tuning (default: WLBSL ISLR)
#
# Optional env overrides:
#   STAGE2_DATASET=CSL_News
#   STAGE3_DATASET=WLBSL
#   STAGE3_TASK=ISLR      # or SLT
#   GPUS="0,1,2,3,4,5,6,7"

STAGE="${1:-stage2}"
REPO="$HOME/projects/dev/Uni-Sign"
cd "$REPO"

: "${GPUS:=0,1,2,3,4,5,6,7}"
: "${STAGE2_DATASET:=CSL_News}"
: "${STAGE3_DATASET:=WLBSL}"
: "${STAGE3_TASK:=ISLR}"

S1_CKPT="out/stage1_pretraining/best_checkpoint.pth"
S2_CKPT="out/stage2_pretraining/best_checkpoint.pth"

mkdir -p logs out

# ----- Python to use (from your current env) -----
PYBIN="$(command -v python)"
echo "[info] Using python: $PYBIN"

# ----- Ensure PyTorch stack present -----
MISSING=0
"$PYBIN" - <<'PY' || MISSING=1
import importlib
for m in ("torch","torchvision","torchaudio"):
    importlib.import_module(m)
print("pytorch stack OK")
PY

if [ "$MISSING" = "1" ]; then
  echo "[info] Installing torch/vision/audio (CUDA 12.1 wheels)..."
  "$PYBIN" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
fi

# ----- Install other reqs (skip torch*) -----
if [ -f requirements.txt ]; then
  grep -viE '^(torch|torchvision|torchaudio)\b' requirements.txt > /tmp/req-no-torch.txt || true
  if [ -s /tmp/req-no-torch.txt ]; then
    "$PYBIN" -m pip install -r /tmp/req-no-torch.txt
  fi
fi

# ----- Sanity checks -----
"$PYBIN" - <<'PY'
import torch
print("torch:", torch.__version__)
assert torch.cuda.is_available(), "CUDA not available"
print("gpu count:", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "No GPUs visible"
PY
nvidia-smi || true

# ----- Multi-GPU env -----
export CUDA_VISIBLE_DEVICES="${GPUS}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

# ----- Launcher -----
if command -v deepspeed >/dev/null 2>&1; then
  INC="localhost:$(echo "$GPUS" | sed 's/,/,localhost:/g')"
  LAUNCH=( deepspeed --include "$INC" --master_port 29511 )
else
  NPROC=$(( $(echo "$GPUS" | tr -cd ',' | wc -c) + 1 ))
  LAUNCH=( torchrun --standalone --nproc_per_node="$NPROC" )
fi
echo "[info] Launcher: ${LAUNCH[*]} (GPUS=$GPUS)"

# ----- Check Stage-1 ckpt -----
if [ ! -f "$S1_CKPT" ]; then
  echo "[error] Missing Stage-1 checkpoint: $S1_CKPT"
  exit 1
fi

case "$STAGE" in
  stage2)
    if [ ! -f pre_training.py ]; then
      echo "[warn] pre_training.py not found; run stage3 instead."
      exit 2
    fi
    mkdir -p out/stage2_pretraining
    LOG=logs/stage2_pretrain.out
    echo "[info] Starting Stage 2 (dataset=${STAGE2_DATASET})..."
    nohup "${LAUNCH[@]}" "$PYBIN" pre_training.py \
      --batch-size 4 \
      --gradient-accumulation-steps 8 \
      --epochs 5 \
      --opt AdamW \
      --lr 3e-4 \
      --quick_break 2048 \
      --output_dir out/stage2_pretraining \
      --finetune "$S1_CKPT" \
      --dataset "$STAGE2_DATASET" \
      --rgb_support \
      > "$LOG" 2>&1 &
    ;;

  stage3)
    CKPT="$S1_CKPT"
    if [ -f "$S2_CKPT" ]; then CKPT="$S2_CKPT"; fi
    mkdir -p out/stage3_finetuning
    LOG=logs/stage3_finetune.out
    echo "[info] Starting Stage 3 (dataset=${STAGE3_DATASET}, task=${STAGE3_TASK})..."
    nohup "${LAUNCH[@]}" "$PYBIN" fine_tuning.py \
      --batch-size 8 \
      --gradient-accumulation-steps 1 \
      --epochs 20 \
      --opt AdamW \
      --lr 3e-4 \
      --output_dir out/stage3_finetuning \
      --finetune "$CKPT" \
      --dataset "$STAGE3_DATASET" \
      --task "$STAGE3_TASK" \
      --max_length 64 \
      --rgb_support \
      > "$LOG" 2>&1 &
    ;;

  *)
    echo "[error] Unknown stage '$STAGE' (use stage2 or stage3)"; exit 1;;
esac

sleep 2
echo "[info] Tailing $LOG ..."
tail -n 200 -f "$LOG"
