#!/usr/bin/env bash
set -Eeuo pipefail

echo "[patch] moving to /var/www/html/Uni-Sign ..."
cd /var/www/html/Uni-Sign

echo "[patch] backing up fine_tuning.py ..."
cp fine_tuning.py fine_tuning.py.bak.$(date +%s)

echo "[patch] applying cleanup + fixes to fine_tuning.py ..."
python - <<'PY'
from pathlib import Path, re
p = Path("fine_tuning.py")
s = p.read_text()

# --- 0) Add a tiny start banner after first imports (helps confirm main runs) ---
if ">>> Uni-Sign fine_tuning starting <<<" not in s:
    s = s.replace("import torch", "import torch\nprint('>>> Uni-Sign fine_tuning starting <<<')", 1)

# --- 1) Drop any old/duplicated defs that kept sneaking back in ---
s = re.sub(r"^def\s+_normalize_tgt_input\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)
s = re.sub(r"^def\s+train_one_epoch\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)
s = re.sub(r"^def\s+evaluate\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)

# Also remove any stray duplicated 'metric_logger.update(loss=loss_value)' blocks that ended up outside functions
s = re.sub(r"\n\s*metric_logger\.update\(loss=loss_value\).*?return\s+\{k:\s*meter\.global_avg.*?\}\s*", "\n", s, flags=re.S)

# --- 2) Ensure we have a sane src normalizer (idempotent) ---
if "_normalize_src_input" not in s:
    s = s.replace("import torch", "import torch\n\n"
        "# --- Batch/input normalization shim ---\n"
        "def _normalize_src_input(src):\n"
        "    import numpy as _np\n"
        "    if isinstance(src, dict):\n"
        "        return src\n"
        "    if isinstance(src, (list, tuple)):\n"
        "        for item in src:\n"
        "            if isinstance(item, dict):\n"
        "                return item\n"
        "        return {'pose': src[0]}\n"
        "    try:\n"
        "        if isinstance(src, _np.ndarray):\n"
        "            src = torch.from_numpy(src)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return {'pose': src}\n"
        "# --- end shim ---\n", 1)

# --- 3) Inject clean ISLR-only tgt normalizer + clean train/eval ---
NEW_FUNCS = r"""
# --- ISLR-only target normalization ---
def _normalize_islr_target(tgt):
    import torch, numpy as _np
    KEYS = ('label','labels','target','targets','y','cls','class_id','id','tgt')
    if isinstance(tgt, dict):
        for k in KEYS:
            if k in tgt:
                return _normalize_islr_target(tgt[k])
        if len(tgt) == 1:
            return _normalize_islr_target(next(iter(tgt.values())))
        raise ValueError(f"ISLR labels not found in dict keys={list(tgt.keys())}. Expected one of {KEYS}.")
    if isinstance(tgt, torch.Tensor):
        return tgt.long().view(-1)
    if isinstance(tgt, _np.ndarray):
        return torch.as_tensor(tgt, dtype=torch.long).view(-1)
    if isinstance(tgt, (list, tuple)):
        vals = []
        for e in tgt:
            t = _normalize_islr_target(e)
            if isinstance(t, torch.Tensor):
                vals.extend(t.view(-1).tolist())
            else:
                vals.append(int(t))
        return torch.tensor(vals, dtype=torch.long).view(-1)
    if isinstance(tgt, str):
        return torch.tensor([int(tgt)], dtype=torch.long)
    try:
        return torch.tensor([int(tgt)], dtype=torch.long)
    except Exception as e:
        raise TypeError(f"Unsupported ISLR label type: {type(tgt)}") from e


def train_one_epoch(args, model, data_loader, optimizer, epoch):
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

        # Normalize tgt per task
        if args.task == "ISLR":
            tgt_input = _normalize_islr_target(tgt_input).cuda(non_blocking=True)
        else:
            if not isinstance(tgt_input, dict):
                tgt_input = {'gt_sentence': tgt_input}
            if args.task == "CSLR" and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']

        # Fwd/Bwd
        out = model(src_input, tgt_input)
        total_loss = out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not (loss_value == loss_value and abs(loss_value) != float("inf")):
            print(f"Loss is {loss_value}, stopping training")
            import sys; sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    print_freq = 10

    use_bf16 = hasattr(model, 'bfloat16_enabled') and model.bfloat16_enabled()
    target_dtype = torch.bfloat16 if use_bf16 else None

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Unpack
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    src_input, tgt_input, meta = batch[:3]
                elif len(batch) == 2:
                    src_input, tgt_input = batch
                    meta = None
                else:
                    raise ValueError(f'Unexpected batch length: {len(batch)}')
            elif isinstance(batch, dict):
                src_input = batch.get('src_input') or batch.get('src')
                tgt_input = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
                meta = batch.get('meta', None)
            else:
                raise ValueError(f'Unexpected batch type: {type(batch)}')

            # Normalize/move src
            src_input = _normalize_src_input(src_input)
            for k, v in list(src_input.items()):
                if isinstance(v, torch.Tensor):
                    if target_dtype is not None and v.dtype.is_floating_point:
                        v = v.to(target_dtype)
                    src_input[k] = v.cuda(non_blocking=True)

            # Ensure dict-style target for refs
            if not isinstance(tgt_input, dict):
                tgt_input = {'gt_sentence': tgt_input}
            if args.task == 'CSLR' and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']

            out = model(src_input, tgt_input)
            total_loss = out['loss']
            metric_logger.update(loss=total_loss.item())

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

    if args.dataset == 'CSL_Daily' and args.task == 'SLT':
        tgt_pres = [' '.join(list(r.replace(' ','').replace('\\n',''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace('，', ',').replace('？','?').replace(' ',''))) for r in tgt_refs]

    if args.task == 'SLT':
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k, v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    elif args.task == 'ISLR':
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
    elif args.task == 'CSLR':
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        for k, v in wer_results.items():
            metric_logger.meters[k].update(v)

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'{phase}_tmp_pres.txt').write_text('\\n'.join(map(str, tgt_pres)) + '\\n')
        (out_dir / f'{phase}_tmp_refs.txt').write_text('\\n'.join(map(str, tgt_refs)) + '\\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""

# Insert NEW_FUNCS once, right before the main-guard (or at end if none)
m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", s, flags=re.M)
if m:
    s = s[:m.start()] + "\n" + NEW_FUNCS + "\n" + s[m.start():]
else:
    s = s + "\n" + NEW_FUNCS + "\n"

# --- 4) Ensure a single, correct main-guard exists ---
# Remove ALL existing main-guards to avoid duplicates
s = re.sub(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*.*\Z", "", s, flags=re.S|re.M)

MAIN_GUARD = r"""
if __name__ == '__main__':
    import os, argparse
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
"""
s = s.rstrip() + "\n" + MAIN_GUARD + "\n"

p.write_text(s)
print("fine_tuning.py cleaned and patched: unique main-guard + clean train/eval + ISLR-only label normalization.")
PY

echo "[patch] done."

