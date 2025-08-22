#!/usr/bin/env python3
"""Create WLBSL train/dev/test symlink splits from a CSV."""
import os, csv, argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="./data/WLBSL/WLBSL_Labels.csv")
ap.add_argument("--rgb-root", default="./dataset/WLBSL/rgb_format")
ap.add_argument("--split-col", default=None, help="split column name if present (e.g., split/phase/subset)")
args = ap.parse_args()

csv_path = Path(os.path.expanduser(args.csv))
rgb_root = Path(os.path.expanduser(args.rgb_root))

# canonical target directory
target_root = Path("./dataset/WLBSL/rgb_format").resolve()
for s in ["train", "dev", "test"]:
    (target_root / s).mkdir(parents=True, exist_ok=True)

# map existing files under rgb_root by lowercase basename for quick lookup
existing = {}
if rgb_root.exists():
    for p in list(rgb_root.glob("*.mp4")) + list(rgb_root.glob("*.MP4")):
        existing[p.name.lower()] = p.resolve()

# normalisation helpers
def norm_path(p: str) -> str:
    p = os.path.expanduser(p)
    return p.replace("/projects/dev/inverse/", "/projects/dev/Uni-Sign/")

def norm_split(x):
    if not isinstance(x, str):
        return "train"
    x = x.strip().lower()
    mapping = {"val": "dev", "valid": "dev", "validation": "dev"}
    x = mapping.get(x, x)
    return x if x in {"train", "dev", "test"} else "train"

video_keys = ["video_path", "video", "filename", "file", "path"]
label_keys = ["label", "gloss"]
split_keys = ["split", "phase", "subset", "partition", "set"]

counts = {"train": 0, "dev": 0, "test": 0}
missing = []

with csv_path.open(newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        row = {k.strip().lower(): (v.strip().replace("\r", "") if isinstance(v, str) else v) for k, v in row.items()}
        vid = next((row.get(k) for k in video_keys if row.get(k)), "")
        if not vid:
            continue
        vid_norm = norm_path(vid)
        base = os.path.basename(vid_norm)
        src = existing.get(base.lower())
        if src is None:
            cand = Path(vid_norm)
            if cand.exists():
                src = cand.resolve()
        if src is None:
            missing.append(vid)
            continue
        if args.split_col:
            sp_val = row.get(args.split_col.strip().lower())
        else:
            sp_val = next((row.get(k) for k in split_keys if row.get(k)), None)
        split = norm_split(sp_val)
        dst = target_root / split / base
        if not dst.exists():
            rel = os.path.relpath(src, dst.parent)
            try:
                os.symlink(rel, dst)
            except FileExistsError:
                pass
        counts[split] += 1

report_path = Path("./out/wlbs_split_report.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("w", encoding="utf-8") as rf:
    for item in missing[:200]:
        rf.write(f"{item}\n")

for s in ["train", "dev", "test"]:
    print(f"{s:<5} : {counts[s]} files")
print(f"missing: {len(missing)} (see {report_path})")

if counts["train"] == 0:
    sys.exit(1)
sys.exit(0)
