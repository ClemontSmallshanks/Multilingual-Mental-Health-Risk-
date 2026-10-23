"""Reusable multi-task trainer (BiLSTM or transformer).

Batches are task-homogeneous (each batch comes from one dataset); batches of all tasks are shuffled together
each epoch. Total loss  L = l_sent * L_sentiment + l_emo * L_emotion + l_risk * L_risk.
Tracks per-task train/val loss + metrics, early-stops on mean validation macro-F1, keeps the best checkpoint.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.config import MAX_LEN, TASK_LABELS
from src.data.datasets import TaskData
from src.models.io import save_checkpoint
from src.models.multitask import task_loss
from src.training.calibration import sigmoid, softmax
from src.training.metrics import compute_task_metrics, primary_score
from src.training.predict import predict_logits


@dataclass
class TrainConfig:
    kind: str = "bilstm"
    epochs: int = 12
    batch_size: int = 32
    lr: float = 2e-3                 # bilstm: all params
    encoder_lr: float = 3e-5         # transformer encoder
    head_lr: float | None = None     # transformer heads; None = legacy lr * 0.5 (what all existing checkpoints used)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    patience: int = 3
    seed: int = 13
    loss_weights: dict = field(default_factory=lambda: {"sentiment": 1.0, "emotion": 1.0, "risk": 1.0})
    max_len: dict = field(default_factory=lambda: dict(MAX_LEN))
    device: str = "cpu"
    smoke: bool = False
    truncation: dict = field(default_factory=dict)  # task -> "head" | "head_tail" (missing = "head")
    head_tokens: int = 64                            # used by head_tail truncation
    amp: bool = False                                # fp16 autocast on CUDA only
    deterministic: bool = False
    save_checkpoint: bool = True                     # False: keep best weights in memory only (experiment runs)
    resume_every_s: int = 600                        # mid-epoch resume snapshot interval (when resume_path is given)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def encode_kwargs(cfg: "TrainConfig", task: str) -> dict:
    return {"truncation": cfg.truncation.get(task, "head"), "head_tokens": cfg.head_tokens}


def class_balance(train: dict[str, TaskData]) -> dict:
    """Imbalance analysis + the weighting decision for each task (written to reports/metrics/class_balance.json)."""
    rep = {}
    for task, d in train.items():
        y = d.labels
        if task == "emotion":
            pos = (y == 1).sum(0); neg = (y == 0).sum(0)
            ratio = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
            pw = np.clip(np.sqrt(ratio), 1.0, 5.0)
            rep[task] = {"strategy": "BCE with pos_weight = clip(sqrt(neg/pos), 1, 5), computed per label over annotated rows only",
                         "positives": pos.tolist(), "negatives": neg.tolist(), "pos_weight": pw.round(3).tolist()}
        else:
            cnt = np.bincount(y, minlength=len(TASK_LABELS[task]))
            imb = float(cnt.max() / max(cnt.min(), 1))
            use = imb > 1.5
            w = (cnt.sum() / (len(cnt) * np.maximum(cnt, 1))) if use else np.ones(len(cnt))
            rep[task] = {"strategy": "inverse-frequency class weights" if use else "none (imbalance ratio <= 1.5; not justified)",
                         "counts": cnt.tolist(), "imbalance_ratio": round(imb, 3), "class_weight": np.round(w, 3).tolist()}
    return rep


def _batches(train: dict[str, TaskData], bs: int, rng: random.Random):
    allb = []
    for task, d in train.items():
        idx = list(range(len(d)))
        rng.shuffle(idx)
        allb += [(task, idx[i : i + bs]) for i in range(0, len(idx), bs)]
    rng.shuffle(allb)
    return allb


def evaluate_split(model, tok, data: dict[str, TaskData], cfg: TrainConfig, thresholds=None):
    """Returns (metrics per task, mean val loss per task). Uses raw (uncalibrated) probabilities."""
    mets, losses = {}, {}
    for task, d in data.items():
        if len(d) == 0:
            continue
        lg = predict_logits(model, tok, d.texts, task, cfg.device, max_len=cfg.max_len[task], amp=cfg.amp, **encode_kwargs(cfg, task))
        y = d.labels
        probs = sigmoid(lg) if task == "emotion" else softmax(lg)
        mets[task] = compute_task_metrics(task, y, probs, None if thresholds is None else thresholds.get(task))
        losses[task] = float(task_loss(task, torch.tensor(lg), torch.tensor(y)).item())
    return mets, losses


def train_multitask(model, tok, train: dict[str, TaskData], val: dict[str, TaskData], cfg: TrainConfig, out_dir, extra_meta=None,
                    resume_path=None) -> dict:
    """`resume_path`: if given, full training state is saved there every `cfg.resume_every_s` seconds and at each
    epoch end, and training resumes from it after a crash/restart (at most ~resume_every_s of work is lost)."""
    set_seed(cfg.seed, cfg.deterministic)
    rng = random.Random(cfg.seed)
    dev = cfg.device
    model.to(dev)
    bal = class_balance(train)
    cw = {t: torch.tensor(bal[t]["class_weight"], dtype=torch.float32, device=dev) for t in train if t != "emotion"}
    pw = torch.tensor(bal["emotion"]["pos_weight"], dtype=torch.float32, device=dev) if "emotion" in train else None

    if cfg.kind == "transformer":
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        groups = [{"params": enc_params, "lr": cfg.encoder_lr}, {"params": head_params, "lr": cfg.head_lr if cfg.head_lr is not None else cfg.lr * 0.5}]
    else:
        groups = [{"params": list(model.parameters()), "lr": cfg.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = sum(-(-len(d) // cfg.batch_size) for d in train.values())
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup_ratio * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    use_amp = cfg.amp and str(dev).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist, best, bad, step = [], -1.0, 0, 0
    best_epoch = 0
    start_epoch, skip, epoch_rng = 1, 0, rng.getstate()
    sums_init = cnts_init = None
    resume = Path(resume_path) if resume_path else None
    if resume is not None and resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); torch.set_rng_state(st["torch_rng"])
        hist, best, bad, step, best_epoch, best_state = st["hist"], st["best"], st["bad"], st["step"], st["best_epoch"], st["best_state"]
        start_epoch, skip, epoch_rng = st["epoch"], st["batch_idx"], st["epoch_rng"]
        sums_init, cnts_init = st["sums"], st["cnts"]
        rng.setstate(epoch_rng)
        print(f"resumed from {resume.name}: epoch {start_epoch}, batch {skip}, step {step}", flush=True)
        if st.get("finished"):
            start_epoch = cfg.epochs + 1  # training had already ended (e.g. early stop) before the restart

    def save_resume(epoch, batch_idx, sums, cnts, finished=False):
        if resume is None:
            return
        tmp = resume.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "hist": hist, "best": best, "bad": bad, "step": step,
                    "best_epoch": best_epoch, "best_state": best_state, "epoch": epoch, "batch_idx": batch_idx,
                    "epoch_rng": epoch_rng, "sums": sums, "cnts": cnts, "finished": finished}, tmp)
        os.replace(tmp, resume)  # atomic: a restart mid-write never leaves a corrupt resume file

    last_save = time.time()
    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_rng = rng.getstate()  # on resume, rng was already reset to the interrupted epoch's start state
        sums = dict(sums_init) if (ep == start_epoch and sums_init is not None) else {t: 0.0 for t in train}
        cnts = dict(cnts_init) if (ep == start_epoch and cnts_init is not None) else {t: 0 for t in train}
        for bi, (task, idx) in enumerate(_batches(train, cfg.batch_size, rng)):
            if ep == start_epoch and bi < skip:
                continue  # already trained on before the restart
            d = train[task]
            texts = [d.texts[i] for i in idx]
            y = torch.tensor(d.labels[idx], device=dev)
            enc = {k: v.to(dev) for k, v in tok.encode(texts, cfg.max_len[task], **encode_kwargs(cfg, task)).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(enc["input_ids"], enc["attention_mask"], task)
            loss = task_loss(task, logits.float(), y, class_weight=cw.get(task), pos_weight=pw if task == "emotion" else None)
            scaler.scale(cfg.loss_weights.get(task, 1.0) * loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            sums[task] += loss.item() * len(idx); cnts[task] += len(idx); step += 1
            if resume is not None and time.time() - last_save > cfg.resume_every_s:
                save_resume(ep, bi + 1, sums, cnts)
                last_save = time.time()
        tr_loss = {t: sums[t] / max(1, cnts[t]) for t in train}
        total_train = sum(cfg.loss_weights.get(t, 1.0) * tr_loss[t] for t in tr_loss)
        vm, vl = evaluate_split(model, tok, val, cfg)
        score = float(np.mean([primary_score(t, m) for t, m in vm.items()]))
        row = {"epoch": ep, "seconds": round(time.time() - t0, 1), "train_loss": {**tr_loss, "total_weighted": total_train},
               "val_loss": {**vl, "total_weighted": sum(cfg.loss_weights.get(t, 1.0) * v for t, v in vl.items())},
               "val": {t: {k: m.get(k) for k in ("accuracy", "macro_f1", "weighted_f1", "pr_auc")} for t, m in vm.items()},
               "val_mean_macro_f1": score, "resumed_within_epoch": bool(ep == start_epoch and skip > 0)}
        hist.append(row)
        print(f"[ep {ep}] {row['seconds']}s train_loss={ {k: round(v,4) for k,v in row['train_loss'].items()} } "
              f"val_macroF1={ {t: round(m['macro_f1'] or 0,4) for t,m in vm.items()} } mean={score:.4f}", flush=True)
        stop = False
        if score > best:
            best, bad = score, 0
            best_epoch = ep
            if cfg.save_checkpoint:
                save_checkpoint(out, model, tok, {"best_epoch": ep, "best_val_mean_macro_f1": score, "smoke": cfg.smoke,
                                                  "train_config": asdict(cfg), **(extra_meta or {})})
            else:
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping at epoch {ep} (best mean val macro-F1 {best:.4f})", flush=True)
                stop = True
        (out / "history.json").write_text(json.dumps({"history": hist, "class_balance": bal, "best_val_mean_macro_f1": best}, indent=2))
        epoch_rng = rng.getstate()  # batch order of the next epoch is drawn from this state
        save_resume(ep + 1, 0, None, None, finished=stop or ep == cfg.epochs)
        last_save = time.time()
        if stop:
            break
    (out / "history.json").write_text(json.dumps({"history": hist, "class_balance": bal, "best_val_mean_macro_f1": best}, indent=2))
    if best_state is not None:  # restore the best-validation weights for evaluation
        model.load_state_dict(best_state)
    return {"history": hist, "best": best, "class_balance": bal, "best_epoch": best_epoch}
