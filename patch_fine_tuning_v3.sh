#!/usr/bin/env bash
set -Eeuo pipefail

echo "[patch] cd /var/www/html/Uni-Sign"
cd /var/www/html/Uni-Sign

echo "[patch] backup fine_tuning.py"
cp fine_tuning.py fine_tuning.py.bak.$(date +%s)

echo "[patch] cleaning + fixing fine_tuning.py"
python - <<'PY'
import re, ast, sys
from pathlib import Path

p = Path("fine_tuning.py")
s = p.read_text()

# 0) Add a start banner once (helps verify main actually runs)
if ">>> Uni-Sign fine_tuning starting <<<" not in s and "import torch" in s:
    s = s.replace("import torch", "import torch\nprint('>>> Uni-Sign fine_tuning starting <<<')", 1)

# 1) Remove any escaped-triple-quote leftovers that cause SyntaxError (\"\"\" -> """)
#    Only fix the exact \"\"\" sequence to avoid messing with normal strings.
s = re.sub(r'\\"\\\"\\\"', '"""', s)  # sometimes tools double-escape in files
s = s.replace(r'\"\"\"', '"""')

# 2) Drop any duplicated/old functions that drifted into the file
for fn in ('_normalize_tgt_input', '_normalize_islr_target', '_as_tgt_dict',
           'train_one_epoch', 'evaluate'):
    s = re.sub(rf'(?ms)^[ \t]*def[ \t]+{fn}\s*\([^)]*\)\s*:[\s\S]*?(?=^[ \t]*def[ \t]+\w+\s*\(|^if __name__ ==[^\n]+$|\Z)',
               '', s)

# 3) Remove orphaned metric_logger fragments (from earlier broken merges)
s = re.sub(r'(?ms)\n\s*metric_logger\.update\(loss=loss_value\)[\s\S]*?return\s+\{k:\s*meter\.global_avg[\s\S]*?\}\s*',
           '\n', s)

# 4) Ensure source normalizer exists (idempotent)
if "_normalize_src_input" not in s:
    src_norm = """
# --- Batch/input normalization shim ---
def _normalize_src_input(src):
    import numpy as _np
    import torch as _t
    # Already a dict of modalities
    if isinstance(src, dict):
        return src
    # Dataloaders sometimes yield (dict, target) or (pose, ...) tuples
    if isinstance(src, (list, tuple)):
        for item in src:
            if isinstance(item, dict):
                return item
        return {'pose': src[0]}
    # Numpy -> Tensor
    try:
        if isinstance(src, _np.ndarray):
            src = _t.from_numpy(src)
    except Exception:
        pass
    return {'pose': src}
# --- end shim ---
"""
    s = s.replace("import torch", "import torch\n" + src_norm, 1)

# 5) Fresh, clean helpers + train/eval (no docstrings -> safe quoting)
NEW_FUNCS = """
def _as_tgt_dict(task, tgt):
    # Ensure target is a dict the model expects.
    # For CSLR, copy gt_gloss -> gt_sentence if missing.
    if isinstance(tgt, dict):
        if task == "CSLR" and 'gt_sentence' not in tgt and 'gt_gloss' in tgt:
            tgt = dict(tgt)
            tgt['gt_sentence'] = tgt['gt_gloss']
        return tgt
    if isinstance(tgt, (list, tuple)):
        for item in tgt:
            if isinstance(item, dict):
                return _as_tgt_dict(task, item)
        return {'gt_sentence': tgt}
    return {'gt_sentence': tgt}


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    import math, torch
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    use_bf16 = hasattr(model, "bfloat16_enabled") and model.bfloat16_enabled()
    target_dtype = torch.bfloat16 if use_bf16 else None

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Unpack
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                src_input, tgt_input, meta = batch[:3]
            elif len(batch) == 2:
                src_input, tgt_input = batch
                meta = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        elif isinstance(batch, dict):
            src_input = batch.get('src_input') or batch.get('src')
            tgt_input = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
            meta = batch.get('meta', None)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Normalize/move src
        src_input = _normalize_src_input(src_input)
        for k, v in list(src_input.items()):
            if isinstance(v, torch.Tensor):
                if target_dtype is not None and v.dtype.is_floating_point:
                    v = v.to(target_dtype)
                src_input[k] = v.cuda(non_blocking=True)

        # Keep tgt as dict (model forward expects dict)
        tgt_input = _as_tgt_dict(args.task, tgt_input)

        # First batch: peek shapes/types
        if step == 0:
            try:
                shapes = {k: (v.shape if isinstance(v, torch.Tensor) else type(v).__name__) for k,v in src_input.items()}
            except Exception:
                shapes = {k: type(v).__name__ for k,v in src_input.items()}
            print("[peek] src_input keys:", list(src_input.keys()))
            print("[peek] src_input shapes/types:", shapes)
            print("[peek] tgt_input keys:", list(tgt_input.keys()))

        out = model(src_input, tgt_input)
        total_loss = out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = float(total_loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping.")
            import sys; sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    import torch
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    use_bf16 = hasattr(model, "bfloat16_enabled") and model.bfloat16_enabled()
    target_dtype = torch.bfloat16 if use_bf16 else None

    with torch.no_grad():
        tgt_pres, tgt_refs = [], []

        for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Unpack
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    src_input, tgt_input, meta = batch[:3]
                elif len(batch) == 2:
                    src_input, tgt_input = batch
                    meta = None
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
            elif isinstance(batch, dict):
                src_input = batch.get('src_input') or batch.get('src')
                tgt_input = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
                meta = batch.get('meta', None)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            # Normalize/move src
            src_input = _normalize_src_input(src_input)
            for k, v in list(src_input.items()):
                if isinstance(v, torch.Tensor):
                    if target_dtype is not None and v.dtype.is_floating_point:
                        v = v.to(target_dtype)
                    src_input[k] = v.cuda(non_blocking=True)

            # Normalize tgt to dict; ensure CSLR gt_sentence
            tgt_input = _as_tgt_dict(args.task, tgt_input)

            out = model(src_input, tgt_input)
            total_loss = out['loss']
            metric_logger.update(loss=total_loss.item())

            # Text generation for metrics (as expected by this project)
            gen_ids = model_without_ddp.generate(out, max_new_tokens=100, num_beams=4)
            id_lists = []
            for g in gen_ids:
                if isinstance(g, torch.Tensor):
                    id_lists.append(g.detach().cpu().tolist())
                else:
                    id_lists.append(list(g))

            tokenizer = model_without_ddp.mt5_tokenizer
            pred_texts = tokenizer.batch_decode(id_lists, skip_special_tokens=True)

            refs = tgt_input.get('gt_sentence')
            if isinstance(refs, (list, tuple)):
                ref_texts = [str(r) for r in refs]
            else:
                ref_texts = [str(refs)] * len(pred_texts)

            tgt_pres.extend(pred_texts)
            tgt_refs.extend(ref_texts)

    # CSL_Daily tokenization quirk
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    # Compute task metrics
    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # Optional debug dumps during --eval
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        from pathlib import Path as _P
        out_dir = _P(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'{phase}_tmp_pres.txt').write_text('\\n'.join(map(str, tgt_pres)) + '\\n')
        (out_dir / f'{phase}_tmp_refs.txt').write_text('\\n'.join(map(str, tgt_refs)) + '\\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""

# Insert NEW_FUNCS once, before main guard if it exists
m = re.search(r'(?m)^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', s)
s = (s[:m.start()] + "\n" + NEW_FUNCS + "\n" + s[m.start():]) if m else (s + "\n" + NEW_FUNCS + "\n")

# 6) Ensure exactly one clean main guard
s = re.sub(r'(?ms)^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*.*\Z', '', s)
MAIN = """
if __name__ == '__main__':
    import os, argparse
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
"""
s = s.rstrip() + "\n" + MAIN + "\n"

# 7) Write & syntax-check
p.write_text(s)
try:
    ast.parse(s)
except SyntaxError as e:
    print(f"[patch ERROR] SyntaxError at line {e.lineno}: {e.msg}")
    # show the bad line for quick debugging
    bad = s.splitlines()[e.lineno-1:e.lineno+2]
    print("Context:\n" + "\n".join(bad))
    sys.exit(1)

print("fine_tuning.py cleaned and patched (quotes fixed; fresh train/eval; safe normalizers).")
PY

echo "[patch] syntax OK and done."

