#!/usr/bin/env bash
# Wrapper to detect WLBSL video root and populate split symlinks.
set -euo pipefail

SRC=$(python3 tools/find_wlbs_videos.py 2>/dev/null || true)
if [ -z "$SRC" ]; then
  echo "Could not locate WLBSL videos" >&2
  exit 1
fi

status=0
if ! python3 script/wlbs_make_splits.py --rgb-root "$SRC" "$@"; then
  status=$?
fi

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
exit $status
