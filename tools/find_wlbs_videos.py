#!/usr/bin/env python3
"""Locate WLBSL RGB video root.

Resolution order:
1. --src /abs/path (validate path contains mp4s)
2. Env vars WLBSL_SRC or WLBSL_VIDEO_SRC
3. Hint file tools/wlbs_src_hint.txt (first non-empty line)
4. Existing repo scan (dataset/, data/, then glob)

On success prints absolute path and exits 0.
On failure prints guidance and exits 2.
"""
import os
import sys
import glob
from pathlib import Path
import argparse


def _count_mp4s(p: Path) -> int:
    if not p.exists() or not p.is_dir():
        return 0
    return len(list(p.glob("*.mp4"))) + len(list(p.glob("*.MP4")))


def _try_path(p: Path) -> bool:
    if _count_mp4s(p) > 0:
        print(str(p.resolve()))
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="absolute path to WLBSL rgb_format directory")
    args = ap.parse_args()

    if args.src:
        src = Path(os.path.expanduser(args.src)).resolve()
        if _try_path(src):
            return 0
        print(f"--src path has no mp4 files: {src}", file=sys.stderr)
        return 2

    env_src = os.environ.get("WLBSL_SRC") or os.environ.get("WLBSL_VIDEO_SRC")
    if env_src:
        src = Path(os.path.expanduser(env_src)).resolve()
        if _try_path(src):
            return 0

    hint_file = Path("tools/wlbs_src_hint.txt")
    if hint_file.exists():
        with hint_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src = Path(os.path.expanduser(line)).resolve()
                if _try_path(src):
                    return 0
                break

    candidates = [
        Path("./dataset/WLBSL/rgb_format"),
        Path("./data/WLBSL/rgb_format"),
    ]
    for cand in candidates:
        if _try_path(cand):
            return 0

    mp4s = glob.glob("**/WLBSL/**/*.mp4", recursive=True) + glob.glob("**/WLBSL/**/*.MP4", recursive=True)
    if mp4s:
        counts: dict[str, int] = {}
        for f in mp4s:
            parent = str(Path(f).resolve().parent)
            counts[parent] = counts.get(parent, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        print(best)
        return 0

    print(
        "Could not locate WLBSL videos.\n"
        "Set an absolute path and re-run, e.g.:\n"
        "  WLBSL_SRC=/home/<user>/.../WLBSL/rgb_format bash script/wlbs_prepare_splits.sh\n"
        "Or write the path into: tools/wlbs_src_hint.txt",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
