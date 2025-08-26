#!/usr/bin/env bash
set -Eeuo pipefail

echo "[patch] cd /var/www/html/Uni-Sign"
cd /var/www/html/Uni-Sign

echo "[patch] backup fine_tuning.py"
cp fine_tuning.py fine_tuning.py.bak.$(date +%s)

echo "[patch] cleaning + fixing fine_tuning.py"
python - <<'PY'
from pathlib import Path, re
p = Path("fine_tuning.py")
s = p.read_text()

# 0) Add a tiny start banner to confirm main executes
if ">>> Uni-Sign fine_tuning starting <<<" not in s:
    s = s.replace("import torch", "import torch\nprint('>>> Uni-Sign fine_tuning starting <<<')", 1)

# 1) Drop any old/duplicated defs that crept in
for fn in (r"_normalize_tgt_input", r"_normalize_islr_target", r"train_one_epoch", r"evaluate"):
    s = re.sub(rf"^def\s+{fn}\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)

# also remove orphaned metric_logger fragments outside functions
s = re.sub(r"\n\s*metric_logger\.update\(loss=loss_value\)[\s\S]*?return\s+\{k:\s*meter\.global_avg[\s\S]*?\}\s*", "\n", s)

# 2) Ensure a sane src normalizer (idempotent)
if "_normalize_src_input" not in s:
    s = s.replace("import torch", "import torch\n\n"
        "# --- Batch/input normalization shim ---\n"
        "def _normalize_src_input(src):\n"
        "    import numpy as _np\n"
        "    # Already a dict of modalities\n"
        "    if isinstance(src, dict):\n"
        "        return src\n"
        "    # Dataloaders sometimes yield (dict, target) or (pose, ...) tuples\n"
        "    if isinstance(src, (list, tuple)):\n"
        "        for item in src:\n"
        "            if isinstance(item, dict):\n"
        "                return item\n"
        "        return {'pose': src[0]}\n"
        "    # Numpy -> Tensor\n"
        "    try:\n"
        "        import torch as _t\n"
        "        if isinstance(src, _np.ndarray):\n"
        "            src = _t.from_numpy(src)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return {'pose': src}\n"
        "# --- end shim ---\n", 1)

NEW_FUNCS = r"""
def _as_tgt_dict(task, tgt):
    \"\"\"Ensure the target is a dict the model expects.
    For ISLR/SLT/CSLR this code keeps/creates 'gt_sentence' when needed.
    We DO NOT coerce to torch tensors here — the model wants a dict.
    \"\"\"
    # Already dict? great.
    if isinstance(tgt, dict):
        # CSLR uses gt_gloss; model reads gt_sentence
        if task == "CSLR" and 'gt_sentence' not in tgt and 'gt_gloss' in tgt:
            tgt = dict(tgt)
            tgt['gt_sentence'] = tgt['gt_gloss']
        return tgt

    # Tuple/list batches where one element is a dict
    if isinstance(tgt, (list, tuple)):
        for item in tgt:
            if isinstance(item, dict):
                return _as_tgt_dict(task, item)
        # Otherwise wrap under gt_sentence (text label name)
        return {'gt_sentence': tgt}

    # Scalars/strings -> wrap
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
        # Unpack dataloader outputs
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                src_input, tgt_input, meta = batch[:3]
            elif len(batch) == 2:
                src_input, tgt_input = batch
                meta = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        elif isinstance(batch, dict):
            src_input  = batch.get('src_input') or batch.get('src')
            tgt_input  = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
            meta       = batch.get('meta', None)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Normalize/move src
        src_input = _normalize_src_input(src_input)
        for k, v in list(src_input.items()):
            if isinstance(v, torch.Tensor):
                if target_dtype is not None and v.dtype.is_floating_point:
                    v = v.to(target_dtype)
                src_input[k] = v.cuda(non_blocking=True)

        # Keep tgt as dict (model expects e.g. 'gt_sentence')
        tgt_input = _as_tgt_dict(args.task, tgt_input)

        # First batch: quick peek of shapes/types
        if step == 0:
            try:
                import torch as _t
                shapes = {k: (v.shape if isinstance(v, _t.Tensor) else type(v).__name__) for k,v in src_input.items()}
            except Exception:
                shapes = {k: type(v).__name__ for k,v in src_input.items()}
            print("[peek] src_input keys:", list(src_input.keys()))
            print("[peek] src_input shapes/types:", shapes)
            print("[peek] tgt_input keys:", list(tgt_input.keys()))

        # Forward/backward with DeepSpeed engine
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
                src_input  = batch.get('src_input') or batch.get('src')
                tgt_input  = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
                meta       = batch.get('meta', None)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            # Normalize/move src
            src_input = _normalize_src_input(src_input)
            for k, v in list(src_input.items()):
                if isinstance(v, torch.Tensor):
                    if target_dtype is not None and v.dtype.is_floating_point:
                        v = v.to(target_dtype)
                    src_input[k] = v.cuda(non_blocking=True)

            # Normalize tgt to dict; ensure cslr gt_sentence
            tgt_input = _as_tgt_dict(args.task, tgt_input)

            out = model(src_input, tgt_input)
            total_loss = out['loss']
            metric_logger.update(loss=total_loss.item())

            # Text generation for metrics
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
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
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
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'{phase}_tmp_pres.txt').write_text('\\n'.join(map(str, tgt_pres)) + '\\n')
        (out_dir / f'{phase}_tmp_refs.txt').write_text('\\n'.join(map(str, tgt_refs)) + '\\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""

# insert our fresh functions before the main guard (or at end)
m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", s, flags=re.M)
s = (s[:m.start()] + "\n" + NEW_FUNCS + "\n" + s[m.start():]) if m else (s + "\n" + NEW_FUNCS + "\n")

# 3) Ensure a single clean main guard
s = re.sub(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*.*\Z", "", s, flags=re.S|re.M)
MAIN = r"""
if __name__ == '__main__':
    import os, argparse
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
"""
p.write_text(s.rstrip() + "\n" + MAIN + "\n")
print("fine_tuning.py cleaned and patched (keep ISLR targets as dict; normalize/move src; fresh train/eval).")
PY

echo "[patch] done."

