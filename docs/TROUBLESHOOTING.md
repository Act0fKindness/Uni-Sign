# Troubleshooting

1. Recap project layout:
   ```bash
   python3 tools/recap_project_map.py
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
     --allow_partial_load auto \
     --epochs 1 --batch-size 8 --num_workers 8
   ```
