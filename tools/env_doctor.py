#!/usr/bin/env python3
"""Environment & dataset diagnostic script."""
import argparse, json, os, sys, subprocess, shutil, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def env_section():
    info = {
        "python": sys.version.split()[0],
        "cuda": run("nvcc --version"),
    }
    try:
        import torch, torchvision, deepspeed
        info.update({
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "deepspeed": getattr(deepspeed, "__version__", ""),
            "gpus": torch.cuda.device_count(),
        })
    except Exception as e:
        info["torch"] = f"not available ({e})"
    info["ffmpeg"] = run("ffmpeg -version")
    return info


def dataset_section(src):
    data = {"src": src, "counts": {}, "broken_symlinks": []}
    for split in ["train", "dev", "test"]:
        p = Path(src) / split
        if p.exists():
            files = list(p.rglob("*.mp4"))
            data["counts"][split] = len(files)
            for f in files:
                if f.is_symlink() and not f.exists():
                    data["broken_symlinks"].append(str(f))
        else:
            data["counts"][split] = 0
    csv = ROOT / "data" / "WLBSL" / "WLBSL_Labels.csv"
    data["csv_exists"] = csv.exists()
    if csv.exists():
        import csv as _csv
        with open(csv) as f:
            header = next(_csv.reader(f))
            data["csv_columns"] = header
    return data


def dataset_smoke(src):
    try:
        sys.path.insert(0, str(ROOT))
        from config import train_label_paths
        from types import SimpleNamespace
        from datasets import S2T_Dataset
        args = SimpleNamespace(dataset='WLBSL', rgb_support=True, max_length=64)
        ds = S2T_Dataset(path=train_label_paths['WLBSL'], args=args, phase='train')
        if len(ds) > 0:
            idx = random.sample(range(len(ds)), min(3, len(ds)))
            items = []
            for i in idx:
                try:
                    it = ds[i]
                    items.append(str(type(it)))
                except Exception as e:
                    items.append(f"error: {e}")
            return {"samples": items}
        return {"samples": []}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./dataset/WLBSL/rgb_format")
    ap.add_argument("--finetune", default="")
    args = ap.parse_args()

    report = {
        "environment": env_section(),
        "dataset": dataset_section(args.src),
        "smoke": dataset_smoke(args.src),
    }
    if args.finetune and Path(args.finetune).exists():
        report["finetune"] = args.finetune
    json_path = OUT / "doctor_report.json"
    md_path = OUT / "doctor_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_lines = ["# Doctor Report", "", "## Environment"]
    for k, v in report["environment"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("\n## Dataset")
    for k, v in report["dataset"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("\n## Smoke Test")
    md_lines.append(str(report.get("smoke")))
    md_path.write_text("\n".join(md_lines))
    print(md_path.read_text())

if __name__ == "__main__":
    main()
