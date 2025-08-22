#!/usr/bin/env python3
"""Peek a sample from WLBSL dataset ensuring repo-local imports."""
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import utils as _utils  # noqa: E402
except Exception as e:  # pragma: no cover - handled in CI
    print(f"skipped (failed to import repo utils: {e})")
    raise SystemExit(0)
assert str(Path(_utils.__file__).resolve()).startswith(str(ROOT)), (
    f"Wrong utils imported: {_utils.__file__}"
)

from config import train_label_paths, dev_label_paths, test_label_paths  # noqa: E402
from datasets import S2T_Dataset  # noqa: E402


def shape_of(x):
    for attr in ("shape", "size"):
        try:
            s = getattr(x, attr)
            return tuple(s) if not callable(s) else tuple(s())
        except Exception:
            pass
    try:  # numpy fallback
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.shape
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WLBSL")
    ap.add_argument("--phase", choices=["train", "dev", "test"], default="train")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--rgb_support", action="store_true")
    args = ap.parse_args()

    ds_name = args.dataset
    split_map = {"train": train_label_paths, "dev": dev_label_paths, "test": test_label_paths}
    label_path = split_map[args.phase][ds_name]

    args_ns = SimpleNamespace(dataset=ds_name, max_length=args.max_length, rgb_support=args.rgb_support)
    ds = S2T_Dataset(path=label_path, args=args_ns, phase=args.phase)
    if len(ds) == 0:
        print("dataset empty")
        return

    sample = ds[0]
    if isinstance(sample, tuple):
        if len(sample) == 2:
            src, tgt = sample
            meta = None
        elif len(sample) == 3:
            src, tgt, meta = sample
        else:
            raise RuntimeError(f"Unexpected sample len={len(sample)}")
    else:
        raise RuntimeError("Dataset __getitem__ should return a tuple")

    print("[peek]")
    print("  len:", len(ds))
    print("  src shape:", shape_of(src))
    print(
        "  tgt type:",
        type(tgt).__name__,
        "meta keys:" if meta else "meta:",
        None if meta is None else getattr(meta, "keys", lambda: [])(),
    )


if __name__ == "__main__":
    main()
