#!/usr/bin/env python3
import os, csv, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="./data/WLBSL/WLBSL_Labels.csv")
ap.add_argument("--rgb-root", default="./dataset/WLBSL/rgb_format")
ap.add_argument("--split-col", default=None, help="split column name if present (e.g., split/phase/subset)")
args = ap.parse_args()

root = Path(args.rgb_root)
for s in ["train","dev","test"]:
    (root / s).mkdir(parents=True, exist_ok=True)

def norm_split(x):
    if not isinstance(x,str): return "train"
    x = x.strip().lower()
    if x in {"val","valid","validation"}: return "dev"
    return x if x in {"train","dev","test"} else "train"

with open(args.csv, newline='', encoding='utf-8') as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        vid = (row.get("video_path") or row.get("video") or row.get("filename") or "").strip()
        if not vid: continue
        base = os.path.basename(os.path.expanduser(vid))
        split = norm_split(row.get(args.split_col) if args.split_col else None)
        src = root / base if (root / base).exists() else Path(os.path.expanduser(vid))
        dst = root / split / base
        if src.exists() and not dst.exists():
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
