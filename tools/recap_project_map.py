#!/usr/bin/env python3
"""Summarize project_map.md into grouped sections with optional symlink following."""
from pathlib import Path
import argparse
import os
import re
import sys

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

PROJECT_MAP = ROOT / "project_map.md"
OUT_DOC = ROOT / "docs" / "project_map_recap.md"


def dir_stats(p: Path, follow=False):
    count = 0
    size = 0
    for root, dirs, files in os.walk(p, followlinks=follow):
        count += len(files)
        for f in files:
            fp = Path(root) / f
            try:
                size += fp.stat().st_size
            except OSError:
                pass
    return count, size


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def collect(follow_links=False):
    if not PROJECT_MAP.exists():
        raise SystemExit(f"project_map.md not found at {PROJECT_MAP}")
    text = PROJECT_MAP.read_text().splitlines()
    entry_points, data_dirs, checkpoints, utilities = [], [], [], []
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        clean = re.sub(r'^[-*]\s*', '', line)
        # Entry points
        if re.search(r"pre_training\.py|fine_tuning\.py|demo/", clean):
            entry_points.append(clean)
        # Data directories
        if re.search(r"data/|dataset/", clean) and (
            "rgb_format" in clean or "pose_format" in clean or "Labels" in clean
        ):
            path = re.sub(r"\s+—.*", "", clean).strip().strip('"')
            dir_path = ROOT / path
            follow = follow_links or (
                "dataset/" in path and ("rgb_format" in path or "pose_format" in path)
            )
            cnt, sz = dir_stats(dir_path, follow=follow)
            data_dirs.append(f"{path} — {cnt} files, {human(sz)}")
        # Checkpoints
        if re.search(r"out/|checkpoints", clean):
            checkpoints.append(clean)
        # Utilities/scripts
        if re.search(r"tools/|script/", clean):
            utilities.append(clean)
    return {
        "Key Entry Points": entry_points,
        "Data Directories": data_dirs,
        "Checkpoints": checkpoints,
        "Utilities & Scripts": utilities,
    }


def format_section(title, items):
    out = [f"## {title}"]
    for it in sorted(set(items)):
        out.append(f"- {it}")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow-symlinks", action="store_true", dest="follow")
    args = ap.parse_args()
    info = collect(follow_links=args.follow)
    lines = ["# Project Map Recap", ""]
    for title, items in info.items():
        lines.append(format_section(title, items))
    output = "\n".join(lines)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
