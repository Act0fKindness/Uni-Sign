#!/usr/bin/env bash
# Wrapper to detect WLBSL video root and populate split symlinks.
set -euo pipefail

SRC=""
if [ "${1:-}" = "--src" ] && [ -n "${2:-}" ]; then
  SRC="$2"
  shift 2
fi

if [ -z "$SRC" ] && [ -n "${WLBSL_SRC:-}" ]; then
  SRC="$WLBSL_SRC"
fi
if [ -z "$SRC" ] && [ -f tools/wlbs_src_hint.txt ]; then
  SRC="$(head -n1 tools/wlbs_src_hint.txt)"
fi
if [ -z "$SRC" ]; then
  SRC="$(python3 tools/find_wlbs_videos.py 2>/dev/null || true)"
fi
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "Could not locate WLBSL videos. Set WLBSL_SRC=/abs/path or edit tools/wlbs_src_hint.txt" >&2
  exit 2
fi

CSV="./data/WLBSL/WLBSL_Labels.csv"
if [ ! -f "$CSV" ]; then
  echo "Missing labels CSV at $CSV" >&2
  exit 2
fi

python3 script/wlbs_make_splits.py --csv "$CSV" --rgb-root "$SRC" --split-col split || true

echo "[counts]"
train_count=0
for p in train dev test; do
  c=$(find ./dataset/WLBSL/rgb_format/$p -type f \( -name '*.mp4' -o -name '*.MP4' \) 2>/dev/null | wc -l)
  printf "  %-5s : %s files\n" "$p" "$c"
  if [ "$p" = "train" ]; then
    train_count=$c
  fi
done

if [ "$train_count" -eq 0 ]; then
  echo "train split empty" >&2
  exit 1
fi

exit 0
