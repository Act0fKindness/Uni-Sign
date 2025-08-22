#!/usr/bin/env python3
"""Scan repository for hardcoded paths and deprecated usages.
Outputs JSON and Markdown reports under out/.
"""
import os, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
JSON_OUT = OUT_DIR / "repo_static_scan.json"
MD_OUT = OUT_DIR / "repo_static_scan.md"

PATTERNS = {
    r"WLBSL": "Hardcoded WLBSL reference",
    r"rgb_format": "WLBSL rgb path",
    r"pose_format": "WLBSL pose path",
    r"WLBSL_Labels\.csv": "Labels CSV",
    r"train_label_paths": "Dataset split path",
    r"dev_label_paths": "Dataset split path",
    r"test_label_paths": "Dataset split path",
    r"\.\/dataset": "./dataset relative path",
    r"\.\/data": "./data relative path",
    r"script/": "script/ path usage",
    r"tools/": "tools/ path usage",
    r"strict=True": "Strict checkpoint load",
    r"pretrained=True": "Deprecated torchvision pretrained flag",
}

RECOMMEND = {
    "pretrained=True": "replace with weights=EfficientNet_B0_Weights.DEFAULT",
    "strict=True": "consider allow_partial_load",
}


def scan_file(path):
    matches = {}
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            for pat, desc in PATTERNS.items():
                if re.search(pat, line):
                    matches.setdefault(pat, []).append({"line": i, "content": line.strip()})
    return matches


def main():
    results = {}
    for root, dirs, files in os.walk(ROOT):
        # skip .git and out
        if any(part.startswith('.') and part != '.' for part in Path(root).parts):
            continue
        for fn in files:
            if fn.endswith(('.py', '.sh')):
                p = Path(root) / fn
                m = scan_file(p)
                if m:
                    results[str(p.relative_to(ROOT))] = m
    OUT_DIR.mkdir(exist_ok=True)
    JSON_OUT.write_text(json.dumps(results, indent=2))

    # human summary
    lines = ["# Repo Static Scan", ""]
    for file, pats in sorted(results.items()):
        lines.append(f"## {file}")
        for pat, instances in pats.items():
            rec = RECOMMEND.get(pat, "")
            lines.append(f"- Pattern `{pat}`: {PATTERNS.get(pat)}" + (f". Recommendation: {rec}" if rec else ""))
            for inst in instances:
                lines.append(f"    - L{inst['line']}: {inst['content']}")
        lines.append("")
    MD_OUT.write_text("\n".join(lines))
    print(MD_OUT.read_text())

if __name__ == "__main__":
    main()
