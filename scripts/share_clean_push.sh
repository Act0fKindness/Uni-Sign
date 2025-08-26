#!/usr/bin/env bash
set -euo pipefail

REPO="/var/www/html/Uni-Sign"
BRANCH="${1:-share/wlbs-islr-fix}"

cd "$REPO"

# Save any local changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "WIP: autosave before cleanup" || true
fi

# Ensure we’re on a shareable branch
git checkout -B "$BRANCH"

# .gitignore additions (idempotent)
add_ignore() { grep -qxF "$1" .gitignore 2>/dev/null || echo "$1" >> .gitignore; }

add_ignore 'data/CSL_News/CSL_News_Labels.csv'
add_ignore 'data/CSL_News/CSL_News_Labels.filtered.skip_pose.json'
add_ignore 'data/**/pose_format/'
add_ignore 'data/**/rgb_format/'
add_ignore 'pretrained_weight/'
add_ignore '*.pth'
add_ignore '*.pt'
add_ignore '*.ckpt'
add_ignore '*.safetensors'
add_ignore '*.mp4'
add_ignore '*.avi'
add_ignore '*.mkv'
add_ignore '*.npz'
add_ignore '*.npy'

git add .gitignore
git commit -m "chore: ignore heavy datasets, weights, and videos" || true

# Untrack large assets (keep them on disk)
git rm --cached -r --ignore-unmatch \
  data/CSL_News/CSL_News_Labels.csv \
  data/CSL_News/CSL_News_Labels.filtered.skip_pose.json \
  pretrained_weight \
  '*.pth' '*.pt' '*.ckpt' '*.safetensors' \
  '*.mp4' '*.avi' '*.mkv' '*.npz' '*.npy' || true
git commit -m "chore: untrack large assets" || true

# Install git-filter-repo if missing
if ! command -v git-filter-repo >/dev/null 2>&1; then
  python -m pip install --user git-filter-repo
  export PATH="$HOME/.local/bin:$PATH"
fi

# Rewrite history to purge the big files
git filter-repo --force --invert-paths \
  --path data/CSL_News/CSL_News_Labels.csv \
  --path data/CSL_News/CSL_News_Labels.filtered.skip_pose.json \
  --path pretrained_weight \
  --path-glob 'data/**/pose_format/**' \
  --path-glob 'data/**/rgb_format/**' \
  --path-glob '*.pth' \
  --path-glob '*.pt' \
  --path-glob '*.ckpt' \
  --path-glob '*.safetensors' \
  --path-glob '*.mp4' \
  --path-glob '*.avi' \
  --path-glob '*.mkv' \
  --path-glob '*.npz' \
  --path-glob '*.npy' || true

# Ensure the origin is the right repo
if ! git remote -v | grep -q 'Act0fKindness/Uni-Sign\.git'; then
  git remote add origin https://github.com/Act0fKindness/Uni-Sign.git
fi

# Push (force, because we rewrote history)
git push -u origin -f "$BRANCH"

echo
echo "✅ Pushed cleaned branch: $BRANCH"
echo "   https://github.com/Act0fKindness/Uni-Sign/tree/$BRANCH"
