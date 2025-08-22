#!/usr/bin/env python3
"""Locate WLBSL RGB video root.

Search order:
1. ./dataset/WLBSL/rgb_format
2. ./data/WLBSL/rgb_format
3. Glob for **/WLBSL/**/*.mp4 and choose the directory with most matches.

Prints absolute path of chosen directory to stdout.
Exit codes:
- 0 success
- 2 no videos found
"""
import sys, os, glob
from pathlib import Path

def count_mp4s(p: Path) -> int:
    if not p.exists() or not p.is_dir():
        return 0
    # case-insensitive mp4
    return len(list(p.glob("*.mp4"))) + len(list(p.glob("*.MP4")))

candidates = [
    Path("./dataset/WLBSL/rgb_format"),
    Path("./data/WLBSL/rgb_format"),
]
for cand in candidates:
    cand = cand.resolve()
    if count_mp4s(cand) > 0:
        print(str(cand))
        sys.exit(0)

# fallback glob search
mp4s = glob.glob("**/WLBSL/**/*.mp4", recursive=True) + glob.glob("**/WLBSL/**/*.MP4", recursive=True)
if not mp4s:
    sys.exit(2)

counts: dict[str, int] = {}
for f in mp4s:
    parent = str(Path(f).resolve().parent)
    counts[parent] = counts.get(parent, 0) + 1
best = max(counts.items(), key=lambda kv: kv[1])[0]
print(best)
sys.exit(0)
