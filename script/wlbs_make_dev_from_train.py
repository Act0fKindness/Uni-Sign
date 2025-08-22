#!/usr/bin/env python3
"""Sample training videos to populate WLBSL dev split."""
from pathlib import Path
import os, glob, random, argparse, sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import utils as _utils  # noqa: E402
except Exception as e:
    print(f"skipped (failed to import repo utils: {e})")
    raise SystemExit(0)
assert str(Path(_utils.__file__).resolve()).startswith(str(ROOT)), (
    f"Wrong utils imported: {_utils.__file__}"
)

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="./dataset/WLBSL/rgb_format/train")
ap.add_argument("--dev",   default="./dataset/WLBSL/rgb_format/dev")
ap.add_argument("--count", type=int, default=500)
ap.add_argument("--seed",  type=int, default=42)
ap.add_argument("--dry-run", action="store_true")
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
        if args.dry_run:
            print(f"would link {dst} -> {target}")
        else:
            os.symlink(target, dst)
            made += 1

# broken symlink check
broken = [f for f in glob.glob(os.path.join(args.dev, "*.mp4")) if os.path.islink(f) and not os.path.exists(os.readlink(f))]
print(f"Created {made} dev symlinks (dev total now: {len([f for f in os.listdir(args.dev) if f.endswith('.mp4')])})")
if broken:
    print(f"Warning: {len(broken)} broken symlinks in dev")
