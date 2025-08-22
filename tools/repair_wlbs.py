#!/usr/bin/env python3
"""Basic repair helper for WLBSL dataset.
Checks split directories and can populate dev split.
"""
from pathlib import Path
import argparse, os, random, shutil, sys

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

from env_doctor import dataset_section, dataset_smoke

def make_dev_from_train(src, count, seed=42, dry_run=False):
    train = Path(src) / "train"
    dev = Path(src) / "dev"
    dev.mkdir(exist_ok=True)
    files = sorted(train.glob("*.mp4"))
    random.seed(seed)
    sample = files if len(files) <= count else random.sample(files, count)
    made = 0
    for f in sample:
        dst = dev / f.name
        if dst.exists():
            continue
        if dry_run:
            print(f"would link {dst} -> {f}")
        else:
            os.symlink(f.resolve(), dst)
            made += 1
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./dataset/WLBSL/rgb_format")
    ap.add_argument("--dev-count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = dataset_section(args.src)
    if info["counts"].get("dev", 0) == 0 and info["counts"].get("train", 0) > 0:
        made = make_dev_from_train(args.src, args.dev_count, args.seed, args.dry_run)
        print(f"Created {made} dev symlinks")
    else:
        print("dev split already populated")

    # After repair, rerun diagnostics
    info = dataset_section(args.src)
    smoke = dataset_smoke(args.src)
    print("Counts:", info["counts"], "broken=", len(info["broken_symlinks"]))
    print("Smoke:", smoke)

if __name__ == "__main__":
    main()
