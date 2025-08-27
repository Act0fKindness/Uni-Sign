set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cp fine_tuning.py "fine_tuning.py.bak.$(date +%s)"

python - <<'PY'
from pathlib import Path; import re

p = Path("fine_tuning.py")
s = p.read_text()

# 0) Banner (harmless; lets you see main actually runs)
if ">>> Uni-Sign fine_tuning starting <<<" not in s:
    s = s.replace("import torch", "import torch\nprint('>>> Uni-Sign fine_tuning starting <<<')", 1)

# 1) Remove any old helpers/loops we will replace
for name in (r"_normalize_tgt_input", r"_normalize_islr_target", r"_as_tgt_dict", r"train_one_epoch", r"evaluate"):
    s = re.sub(rf"^def\s+{name}\s*\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)

# 2) Ensure we have a robust src normalizer (idempotent)
if "_normalize_src_input(" not in s:
    s = s.replace(
        "import torch",
        "import torch\n\n"
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
        "            import torch as _t; src = _t.from_numpy(src)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return {'pose': src}\n"
        "# --- end shim ---\n",
        1,
    )

# 3) Insert clean helpers + train/eval
NEW = r"""
# --- Target dict helper (no numeric casting) ---
def _as_tgt_dict(task, tgt):
    """
    Ensure the target is a dict with a 'gt_sentence' (or equivalent) key,
    which is what the model expects for all Stage-3 tasks.
    """
    # If already a dict, copy and normalize key names
    if isinstance(tgt, dict):
        d = dict(tgt)
    else:
        # Wrap non-dict targets as a sentence
        d = {"gt_sentence": str(tgt)}

    # Normalize common variants -> gt_sentence
    if "gt_sentence" not in d:
        if "sentence" in d:
            d["gt_sentence"] = d["sentence"]
        elif "gt_gloss" in d:
            d["gt_sentence"] = d["gt_gloss"]
        elif "gloss" in d:
            d["gt_sentence"] = d["gloss"]
        else:
            # Last resort: stringify the whole tgt
            d["gt_sentence"] = str(next(iter(d.values())))

    return d


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    import math, sys, torch
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Unpack
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                src_input, tgt_input, _meta = batch[:3]
            elif len(batch) == 2:
                src_input, tgt_input = batch
                _meta = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        elif isinstance(batch, dict):
            src_input = batch.get('src_input') or batch.get('src')
            tgt_input = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
            _meta = batch.get('meta', None)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Normalize src and move to GPU (no dtype juggling)
        src_input = _normalize_src_input(src_input)
        for k, v in list(src_input.items()):
            if isinstance(v, torch.Tensor):
                src_input[k] = v.cuda(non_blocking=True)

        # Normalize tgt to dict with gt_sentence
        tgt_input = _as_tgt_dict(args.task, tgt_input)

        # Special-casing CSLR stays harmless via the helper above, but keep this for clarity
        if args.task == "CSLR" and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']

        out = model(src_input, tgt_input)
        total_loss = out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = float(total_loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training"); sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Peek once at step 0
        if step == 0 and utils.is_main_process():
            print('[peek] src_input keys:', list(src_input.keys()))
            if isinstance(tgt_input, dict):
                print('[peek] tgt_input keys:', list(tgt_input.keys()))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    import torch
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    with torch.no_grad():
        tgt_pres, tgt_refs = [], []

        for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Unpack
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    src_input, tgt_input, _meta = batch[:3]
                elif len(batch) == 2:
                    src_input, tgt_input = batch
                    _meta = None
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
            elif isinstance(batch, dict):
                src_input = batch.get('src_input') or batch.get('src')
                tgt_input = batch.get('tgt_input') or batch.get('tgt') or batch.get('label')
                _meta = batch.get('meta', None)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            # Normalize/move
            src_input = _normalize_src_input(src_input)
            for k, v in list(src_input.items()):
                if isinstance(v, torch.Tensor):
                    src_input[k] = v.cuda(non_blocking=True)

            tgt_input = _as_tgt_dict(args.task, tgt_input)
            if args.task == "CSLR" and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']

            out = model(src_input, tgt_input)
            metric_logger.update(loss=float(out['loss'].item()))

            # Generate text ids, decode
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
                # Broadcast a single reference across the batch length (rare)
                ref_texts = [str(refs)] * len(pred_texts)

            tgt_pres.extend(pred_texts)
            tgt_refs.extend(ref_texts)

    # Dataset/task-specific post-processing
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    # Metrics
    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k, v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        for k, v in wer_results.items():
            metric_logger.meters[k].update(v)

    # Optional dump
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        from pathlib import Path
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{phase}_tmp_pres.txt").write_text("\n".join(map(str, tgt_pres)) + "\n")
        (out_dir / f"{phase}_tmp_refs.txt").write_text("\n".join(map(str, tgt_refs)) + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""
# Insert NEW just before the main-guard (or append)
m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", s, flags=re.M)
s = (s[:m.start()] + "\n" + NEW + "\n" + s[m.start():]) if m else (s + "\n" + NEW + "\n")

# 4) Ensure a single, clean main-guard
s = re.sub(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*.*\Z",
           "", s, flags=re.S|re.M)
s = s.rstrip() + """
if __name__ == '__main__':
    import os, argparse
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
""" + "\n"

p.write_text(s)
print("Patched fine_tuning.py with clean tgt handling for Stage-3.")
PY
