#!/usr/bin/env python3
"""Summarize project_map.md into grouped sections.
Parses project_map.md and prints/writes outline of key files.
"""
import re
import os
from pathlib import Path

PROJECT_MAP = Path(__file__).resolve().parent.parent / "project_map.md"
OUT_DOC = Path(__file__).resolve().parent.parent / "docs" / "project_map_recap.md"

def collect():
    if not PROJECT_MAP.exists():
        raise SystemExit(f"project_map.md not found at {PROJECT_MAP}")
    text = PROJECT_MAP.read_text().splitlines()
    entry_points = []
    data_dirs = []
    checkpoints = []
    utilities = []
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # remove bullet prefix
        clean = re.sub(r'^[-*]\s*', '', line)
        # Entry points
        if re.search(r"pre_training\.py|fine_tuning\.py|demo/", clean):
            entry_points.append(clean)
        # Data directories
        if re.search(r"data/|dataset/", clean) and ("rgb_format" in clean or "pose_format" in clean or "Labels" in clean):
            data_dirs.append(clean)
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
    info = collect()
    lines = ["# Project Map Recap", ""]
    for title, items in info.items():
        lines.append(format_section(title, items))
    output = "\n".join(lines)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(output)
    print(output)

if __name__ == "__main__":
    main()
