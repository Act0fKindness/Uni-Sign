# Troubleshooting

1. Recap project layout (follow symlinks for accurate counts):
   ```bash
   python3 tools/recap_project_map.py --follow-symlinks
   ```
2. Environment & dataset doctor:
   ```bash
   bash script/doctor.sh --src "$PWD/dataset/WLBSL/rgb_format" --finetune ./out/stage1_pretraining/best_checkpoint.pth
   ```
3. Attempt auto-repair if issues:
   ```bash
   python3 tools/repair_wlbs.py --src "$PWD/dataset/WLBSL/rgb_format" --dev-count 500
   ```
4. Peek a dataset sample:
   ```bash
   python3 script/wlbs_dataset_peek.py --dataset WLBSL --rgb_support --phase train
   ```
5. Run a training smoke test:
   ```bash
   torchrun --standalone --nproc_per_node=1 fine_tuning.py \
     --task ISLR --dataset WLBSL --rgb_support \
     --output_dir ./out/wlbs_stage2_smoke_rgb \
   --finetune ./out/stage1_pretraining/best_checkpoint.pth \
    --epochs 1 --batch-size 8 --num_workers 8
   ```

## Wrong module imported
If a tool errors with paths outside this repo, ensure the repo root is pinned:

```python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils as _utils
assert str(Path(_utils.__file__).resolve()).startswith(str(ROOT))
```
