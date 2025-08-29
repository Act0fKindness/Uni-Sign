#!/usr/bin/env python3
"""
Project File System Map + Safe Previews

- Prints a directory tree (bounded depth and entries).
- Summarises sizes & counts per directory.
- Embeds safe previews of text-y files (CSV/JSON/MD/TXT/PY, etc.).
- Skips heavy/binary blobs by default (video, model weights, archives, etc.).
- Never loads entire large files; previews first N lines and truncates long lines.

Usage (example):
  python3 tools/project_map.py \
    --root ~/projects/dev/Uni-Sign \
    --out project_map.md \
    --max-depth 4 \
    --max-previews 80 \
    --include-exts .csv,.json,.md,.txt,.py \
    --skip-dirs .git,__pycache__,mmpose.broken-*,dataset/CSL_News/rgb_format \
    --skip-exts .pth,.pt,.mp4,.mkv,.avi,.zip,.gz,.npz,.npy,.onnx,.tar,.7z,.pkl
"""
from __future__ import annotations
import argparse, os, sys, stat, fnmatch, time
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

def human(n: int) -> str:
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024 or u == "PB": return f"{n:.1f} {u}" if u!="B" else f"{n} {u}"
        n /= 1024

def is_probably_text(path: Path, sniff_bytes: int = 2048) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
        if b"\x00" in chunk:  # NUL byte → likely binary
            return False
        # allow high-ASCII; just reject control chars aside from whitespace
        bad_controls = sum(b < 9 or (13 < b < 32) for b in chunk)
        return bad_controls / max(1, len(chunk)) < 0.01
    except Exception:
        return False

def preview_lines(path: Path, max_lines: int, max_chars_per_line: int) -> List[str]:
    out: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                line = line.rstrip("\n")
                if len(line) > max_chars_per_line:
                    line = line[:max_chars_per_line] + " …"
                out.append(line)
    except Exception as e:
        out.append(f"[preview error: {e}]")
    return out

def match_any(patterns: Iterable[str], text: str) -> bool:
    return any(fnmatch.fnmatch(text, p) for p in patterns)

def safe_rel(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)

def walk_limited(root: Path, max_depth: int, skip_dirs: List[str]) -> Iterable[Tuple[int, Path]]:
    """Yield (depth, path) for directories up to max_depth, skipping patterns."""
    stack = [(0, root)]
    while stack:
        depth, d = stack.pop()
        yield depth, d
        if depth >= max_depth: 
            continue
        try:
            entries = sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        except PermissionError:
            continue
        for sub in reversed(entries):
            rel = safe_rel(root, sub)
            if match_any(skip_dirs, rel) or match_any(skip_dirs, sub.name):
                continue
            stack.append((depth+1, sub))

def gather_dir_summary(d: Path, skip_dirs: List[str], skip_exts: List[str]) -> Tuple[int,int]:
    files = 0
    size = 0
    for dirpath, dirnames, filenames in os.walk(d):
        rel = os.path.relpath(dirpath, d)
        if rel == ".": rel = ""
        # prune dirs
        pruned = []
        for name in list(dirnames):
            subrel = os.path.join(rel, name) if rel else name
            if match_any(skip_dirs, subrel) or match_any(skip_dirs, name):
                pruned.append(name)
        for p in pruned:
            dirnames.remove(p)
        # files
        for fn in filenames:
            if any(fn.lower().endswith(ext.lower()) for ext in skip_exts):
                continue
            files += 1
            try:
                size += os.stat(os.path.join(dirpath, fn)).st_size
            except OSError:
                pass
    return files, size

def build_tree_section(root: Path, max_depth: int, skip_dirs: List[str], skip_exts: List[str], max_entries: int) -> str:
    lines = []
    lines.append(f"## Directory Tree (depth ≤ {max_depth})\n")
    def print_dir(depth: int, path: Path):
        files, size = gather_dir_summary(path, skip_dirs, skip_exts)
        indent = "  " * depth
        rel = safe_rel(root, path) or "."
        lines.append(f"{indent}- **{rel}** — {files} files, {human(size)}")

        # list a handful of child entries (files + dirs)
        try:
            entries = sorted(list(path.iterdir()), key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
        except PermissionError:
            entries = []

        shown = 0
        for p in entries:
            relp = safe_rel(root, p)
            if p.is_dir():
                if match_any(skip_dirs, relp) or match_any(skip_dirs, p.name): 
                    continue
                # child dir count shown in its own line when recursing
                continue
            if any(p.name.lower().endswith(ext.lower()) for ext in skip_exts):
                continue
            if shown >= max_entries:
                break
            try:
                sz = human(p.stat().st_size)
            except OSError:
                sz = "?"
            lines.append(f"{indent}  - {p.name} ({sz})")
            shown += 1

    for depth, d in walk_limited(root, max_depth, skip_dirs):
        print_dir(depth, d)

    lines.append("")
    return "\n".join(lines)

def build_previews(root: Path, include_exts: List[str], skip_dirs: List[str],
                   max_previews: int, max_lines: int, max_chars: int) -> str:
    lines = []
    lines.append("## File Previews\n")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel == ".": rel = ""
        # prune dirs
        pruned = []
        for name in list(dirnames):
            subrel = os.path.join(rel, name) if rel else name
            if match_any(skip_dirs, subrel) or match_any(skip_dirs, name):
                pruned.append(name)
        for p in pruned:
            dirnames.remove(p)

        for fn in sorted(filenames):
            if count >= max_previews: 
                lines.append(f"\n> _(stopped after {max_previews} previews)_\n")
                return "\n".join(lines)
            ext = "".join(Path(fn).suffixes).lower()
            if include_exts and not any(ext.endswith(e.lower()) for e in include_exts):
                continue
            path = Path(dirpath) / fn
            try:
                st = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not is_probably_text(path):
                continue
            # write preview block
            preview = preview_lines(path, max_lines=max_lines, max_chars_per_line=max_chars)
            relp = safe_rel(root, path)
            lines.append(f"### `{relp}` ({human(st.st_size)})")
            lang = "csv" if ext.endswith(".csv") else "json" if ext.endswith(".json") else "text"
            lines.append(f"```{lang}")
            lines.extend(preview if preview else ["[empty]"])
            lines.append("```")
            lines.append("")
            count += 1
    if count == 0:
        lines.append("_No previewable files found with current filters._\n")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Build a project file system map with previews")
    ap.add_argument("--root", required=True, help="Project root directory")
    ap.add_argument("--out", default="project_map.md", help="Output Markdown file")
    ap.add_argument("--max-depth", type=int, default=4, help="Max directory depth for tree")
    ap.add_argument("--max-entries-per-dir", type=int, default=8, help="Max file entries shown per directory in tree")
    ap.add_argument("--max-previews", type=int, default=60, help="Max number of file previews")
    ap.add_argument("--max-lines", type=int, default=5, help="Max lines per file preview")
    ap.add_argument("--max-chars", type=int, default=300, help="Max characters per line in preview")
    ap.add_argument("--include-exts", default=".csv,.json,.md,.txt,.py", help="Comma-separated extensions to preview")
    ap.add_argument("--skip-dirs", default=".git,__pycache__,.mypy_cache,.pytest_cache,venv,env,build,dist,*.egg-info,mmpose.broken-*,dataset/CSL_News/rgb_format",
                    help="Comma-separated glob patterns for directories to skip")
    ap.add_argument("--skip-exts", default=".pth,.pt,.mp4,.mkv,.avi,.zip,.gz,.npz,.npy,.onnx,.tar,.7z,.pkl",
                    help="Comma-separated file extensions to skip entirely (for tree/size calc)")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    include_exts = [e.strip() for e in args.include_exts.split(",") if e.strip()]
    skip_dirs    = [s.strip() for s in args.skip_dirs.split(",") if s.strip()]
    skip_exts    = [e.strip() for e in args.skip_exts.split(",") if e.strip()]

    hdr = []
    hdr.append(f"# Project Map — {root.name}")
    hdr.append("")
    hdr.append(f"- **Root:** `{root}`")
    hdr.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hdr.append(f"- **Max depth:** {args.max_depth}, **Max previews:** {args.max_previews}")
    hdr.append(f"- **Previewed extensions:** {', '.join(include_exts)}")
    hdr.append("")

    tree = build_tree_section(root, args.max_depth, skip_dirs, skip_exts, args.max_entries_per_dir)
    previews = build_previews(root, include_exts, skip_dirs, args.max_previews, args.max_lines, args.max_chars)

    out_path = Path(args.out).resolve()
    out_path.write_text("\n".join(hdr) + "\n" + tree + "\n" + previews, encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

