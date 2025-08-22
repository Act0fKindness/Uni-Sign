#!/usr/bin/env python3
import argparse, os, sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from datasets import S2T_Dataset
    from config import train_label_paths, dev_label_paths, test_label_paths
except ModuleNotFoundError as e:
    print(f"skipped (torch not installed): {e}")
    raise SystemExit

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='WLBSL')
ap.add_argument('--rgb_support', action='store_true')
ap.add_argument('--phase', default='train')
ap.add_argument('--max_length', type=int, default=64)
args = ap.parse_args()

args_ns = SimpleNamespace(dataset=args.dataset, rgb_support=args.rgb_support, max_length=args.max_length)
paths = {'train': train_label_paths, 'dev': dev_label_paths, 'test': test_label_paths}
label_path = paths[args.phase][args.dataset]

ds = S2T_Dataset(path=label_path, args=args_ns, phase=args.phase)
print(ds)
if len(ds) == 0:
    print('dataset empty')
else:
    item = ds[0]
    if isinstance(item, tuple):
        print('tuple length:', len(item))
        for idx, part in enumerate(item):
            if hasattr(part, 'shape'):
                print(f' part {idx} type {type(part)} shape {getattr(part, "shape", None)}')
            else:
                print(f' part {idx} type {type(part)}')
    else:
        print('item type:', type(item))
