#!/usr/bin/env python3
import os, glob, random, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="./dataset/WLBSL/rgb_format/train")
ap.add_argument("--dev",   default="./dataset/WLBSL/rgb_format/dev")
ap.add_argument("--count", type=int, default=500)
ap.add_argument("--seed",  type=int, default=42)
args = ap.parse_args()

os.makedirs(args.dev, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.train, "*.mp4")))
random.seed(args.seed)
sample = files if len(files) <= args.count else random.sample(files, args.count)

made = 0
for p in sample:
    target = os.path.realpath(p)
    dst = os.path.join(args.dev, os.path.basename(p))
    if not os.path.exists(dst):
        os.symlink(target, dst)
        made += 1
print(f"Created {made} dev symlinks (dev total now: {len([f for f in os.listdir(args.dev) if f.endswith('.mp4')])})")
