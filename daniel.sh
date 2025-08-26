cd /var/www/html/Uni-Sign

cp fine_tuning.py fine_tuning.py.bak.$(date +%s)

python - <<'PY'
from pathlib import Path, re
p = Path("fine_tuning.py")
s = p.read_text()

# 0) Make sure we don't have any leftover debug/imports that shadow torch (optional, defensive)
s = s.replace("import torch as _tpeek", "import torch as _tpeek  # safe alias")  # harmless if absent

# 1) Drop ALL old definitions of _normalize_tgt_input / train_one_epoch / evaluate
s = re.sub(r"^def\s+_normalize_tgt_input\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)
s = re.sub(r"^def\s+train_one_epoch\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)
s = re.sub(r"^def\s+evaluate\([^\n]*\):.*?(?=^def\s|\Z)", "", s, flags=re.S|re.M)

# Also remove any duplicated garbled blocks that start with 'metric_logger.update(loss=loss_value)' dangling after eval
s = re.sub(r"\n\s*metric_logger\.update\(loss=loss_value\).*?return\s+\{k: meter\.global_avg.*?\}\s*\n", "\n", s, flags=re.S)

# 2) Ensure we have the src normalizer; if not found, add a simple one (most users already have it)
if "_normalize_src_input" not in s:
    s = s.replace("import torch", "import torch\n\n# --- Batch/input normalization shim ---\n"
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

# 3) Inject a *new* ISLR-only target normalizer + clean train_one_epoch + clean evaluate
NEW_FUNCS = r"""
# --- ISLR-only target normalization ---
def _normalize_islr_target(tgt):
    import numpy as _np
    KEYS = ('label','labels','target','targets','y','cls','class_id','id','tgt')

    # Drill into dicts to find a label field
    if isinstance(tgt, dict):
        for k in KEYS:
            if k in tgt:
                return _normalize_islr_target(tgt[k])
        if len(tgt) == 1:
            return _normalize_islr_target(next(iter(tgt.values())))
        raise ValueError(f"ISLR labels not found in dict keys={list(tgt.keys())}. Expected one of {KEYS}.")

    # Tensor / ndarray -> long vec
    if isinstance(tgt, torch.Tensor):
        return tgt.long().view(-1)
    if 'ndarray' in str(type(tgt)):  # lazy check to avoid importing numpy if not needed
        try:
            return torch.as_tensor(tgt, dtype=torch.long).view(-1)
        except Exception:
            pass

    # Sequence -> list of ints
    if isinstance(tgt, (list, tuple)):
        vals = []
        for e in tgt:
            t = _normalize_islr_target(e)
            if isinstance(t, torch.Tensor):
                vals.extend(t.view(-1).tolist())
            else:
                vals.append(int(t))
        return torch.tensor(vals, dtype=torch.long).view(-1)

    # Strings and scalars -> int tensor of len 1
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
        # Unpack (src, tgt [, meta]) or dict
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

        # Normalize/move src to GPU
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
            # SLT / CSLR keep dict-style targets
            if not isinstance(tgt_input, dict):
                tgt_input = {'gt_sentence': tgt_input}
            if args.task == "CSLR" and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']

        # Forward/backward
        out = model(src_input, tgt_input)
        total_loss = out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    use_bf16 = hasattr(model, "bfloat16_enabled") and model.bfloat16_enabled()
    target_dtype = torch.bfloat16 if use_bf16 else None

    with torch.no_grad():
        tgt_pres = []  # predicted strings
        tgt_refs = []  # reference strings

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

            # Ensure dict-style targets for generation/reference strings
            if not isinstance(tgt_input, dict):
                tgt_input = {'gt_sentence': tgt_input}
            if args.task == "CSLR" and 'gt_sentence' not in tgt_input and 'gt_gloss' in tgt_input:
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']

            # Forward (no backward)
            out = model(src_input, tgt_input)
            total_loss = out['loss']
            metric_logger.update(loss=total_loss.item())

            # Generate token ids and decode
            gen_ids = model_without_ddp.generate(out, max_new_tokens=100, num_beams=4)
            # convert to list-of-ids for tokenizer
            pred_id_lists = []
            for g in gen_ids:
                if isinstance(g, torch.Tensor):
                    pred_id_lists.append(g.detach().cpu().tolist())
                else:
                    pred_id_lists.append(list(g))
            tokenizer = model_without_ddp.mt5_tokenizer
            pred_texts = tokenizer.batch_decode(pred_id_lists, skip_special_tokens=True)

            # refs
            refs = tgt_input.get('gt_sentence')
            if isinstance(refs, (list, tuple)):
                ref_texts = [str(r) for r in refs]
            else:
                ref_texts = [str(refs)] * len(pred_texts)

            # collect
            tgt_pres.extend(pred_texts)
            tgt_refs.extend(ref_texts)

    # Post-process for CSL_Daily SLT quirk
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

    # Optionally dump preds/refs only when evaluating (args.eval on a single process)
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        out_dir = Path(args.output_dir)
        (out_dir / f"{phase}_tmp_pres.txt").write_text("\n".join(map(str, tgt_pres)) + "\n")
        (out_dir / f"{phase}_tmp_refs.txt").write_text("\n".join(map(str, tgt_refs)) + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""

# 4) Insert NEW_FUNCS just before the main guard to ensure defs exist before use
m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", s, flags=re.M)
if m:
    s = s[:m.start()] + "\n" + NEW_FUNCS + "\n" + s[m.start():]
else:
    s = s + "\n" + NEW_FUNCS + "\n"

p.write_text(s)
print("fine_tuning.py patched: ISLR-only target normalizer + clean train/eval.")
PY

