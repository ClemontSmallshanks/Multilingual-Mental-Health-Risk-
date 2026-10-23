"""Config-driven experiment runner (Stage 2 of reports/experiment_plan.md).

One run = one config x one seed, written to  <RUNS_DIR>/<experiment>/seed_<seed>/ :
  config.json          resolved config (incl. TrainConfig)
  env.json             package versions, GPU, git commit
  history.json         per-epoch train/val losses + val metrics
  metrics.json         val metrics (always) and test metrics (only if evaluate_test), calibration, timing, memory
  predictions_<split>_<task>.csv   text_id, language, source, label(s), calibrated per-class probabilities (no text)
A run whose metrics.json exists is skipped (resume after a Colab disconnect).

Test-set discipline: test predictions are produced only when the config sets "evaluate_test": true. Configs that
do this are the pre-registered final evaluations in reports/experiment_plan.md Sec. 7.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import EMOTION_LABELS, MAX_LEN, ROOT, RUNS_DIR, TASK_LABELS
from src.data.datasets import load_all
from src.evaluation.evaluate import _probs
from src.experiments.env import capture
from src.models.io import detect_device, save_checkpoint
from src.models.multitask import MultiTaskTransformer
from src.models.tokenization import HFTokenizer
from src.training.calibration import fit_temperature, sigmoid
from src.training.metrics import compute_task_metrics, tune_emotion_thresholds
from src.training.predict import predict_logits
from src.training.trainer import TrainConfig, encode_kwargs, set_seed, train_multitask

TRAIN_KEYS = {f for f in TrainConfig.__dataclass_fields__} - {"kind", "seed", "device", "smoke"}


def runs_dir() -> Path:
    return Path(os.environ.get("MHRISK_RUNS_DIR", RUNS_DIR))


def expand(config: dict) -> list[dict]:
    """Expand an optional "grid": {train_key: [values...]} into one config per combination (named by its values)."""
    grid = config.get("grid")
    if not grid:
        return [config]
    keys = sorted(grid)
    out = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        c = copy.deepcopy(config)
        c.pop("grid")
        c.setdefault("train", {}).update(dict(zip(keys, vals)))
        c["name"] = config["name"] + "__" + "_".join(f"{k}{v}" for k, v in zip(keys, vals))
        out.append(c)
    return out


def _train_config(cfg: dict, seed: int, device: str) -> TrainConfig:
    t = dict(cfg.get("train", {}))
    bad = set(t) - TRAIN_KEYS
    if bad:
        raise ValueError(f"unknown train keys in config {cfg['name']}: {sorted(bad)}")
    max_len = dict(MAX_LEN)
    max_len.update(t.pop("max_len", {}))
    tc = TrainConfig(kind="transformer", seed=seed, device=device, max_len=max_len, save_checkpoint=False, **t)
    tc.loss_weights = {k: v for k, v in tc.loss_weights.items() if k in cfg["tasks"]}
    return tc


def _predict_split(model, tok, data, tc: TrainConfig, cal: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    mets, rows = {}, {}
    for task, d in data.items():
        lg = predict_logits(model, tok, d.texts, task, tc.device, max_len=tc.max_len[task], amp=tc.amp, **encode_kwargs(tc, task))
        p_cal = _probs(task, lg, cal["temperature"][task])
        thr = np.array(cal["emotion_thresholds"]) if task == "emotion" else None
        m = compute_task_metrics(task, d.labels, p_cal, thr)
        m_raw = compute_task_metrics(task, d.labels, _probs(task, lg), thr)
        m["uncalibrated"] = {k: m_raw.get(k) for k in ("ece", "brier", "macro_f1", "pr_auc", "roc_auc") if k in m_raw}
        m["by_language"] = {l: compute_task_metrics(task, d.labels[d.languages == l], p_cal[d.languages == l], thr) for l in sorted(set(d.languages))}
        src = d.df.source.to_numpy()
        m["by_source"] = {s: compute_task_metrics(task, d.labels[src == s], p_cal[src == s], thr) for s in sorted(set(src))}
        if task == "emotion":
            y = d.labels
            pred = (p_cal >= thr).astype(int)
            valid = y >= 0
            m["per_label_error_rate"] = float(((pred != y) & valid).sum() / valid.sum())
            m["exact_set_mismatch_rate"] = float(np.mean([((a != b) & (b >= 0)).any() for a, b in zip(pred, y)]))
        mets[task] = m
        df = d.df[["text_id", "language", "source"]].copy()
        df["task"] = task
        if task == "emotion":
            for j, e in enumerate(EMOTION_LABELS):
                df[f"y_{e}"] = d.labels[:, j]
                df[f"p_{e}"] = p_cal[:, j].round(6)
        else:
            df["y"] = d.labels
            for j, l in enumerate(TASK_LABELS[task]):
                df[f"p_{l}"] = p_cal[:, j].round(6)
        rows[task] = df
    return mets, rows


def _fit_calibration(model, tok, val, tc: TrainConfig) -> dict:
    cal = {"temperature": {}, "emotion_thresholds": None, "fit_on": "validation split"}
    for task, d in val.items():
        lg = predict_logits(model, tok, d.texts, task, tc.device, max_len=tc.max_len[task], amp=tc.amp, **encode_kwargs(tc, task))
        cal["temperature"][task] = fit_temperature(lg, d.labels, "sigmoid" if task == "emotion" else "softmax")
        if task == "emotion":
            cal["emotion_thresholds"] = tune_emotion_thresholds(d.labels, sigmoid(lg, cal["temperature"][task])).tolist()
    return cal


def run_one(cfg: dict, seed: int, device: str | None = None, evaluate_test: bool | None = None, limit: int | None = None,
            build=None) -> Path:
    """`build(cfg) -> (model, tokenizer)` overrides model construction (used by offline tests)."""
    out = runs_dir() / cfg["name"] / f"seed_{seed}"
    if (out / "metrics.json").exists():
        print(f"skip {cfg['name']} seed {seed} (done)", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    device = detect_device(device or cfg.get("device"))
    eval_test = cfg.get("evaluate_test", False) if evaluate_test is None else evaluate_test
    tc = _train_config(cfg, seed, device)
    tasks = cfg["tasks"]
    (out / "config.json").write_text(json.dumps({**cfg, "seed": seed, "evaluate_test": eval_test, "limit": limit,
                                                 "resolved_train_config": asdict(tc)}, indent=2))
    (out / "env.json").write_text(json.dumps(capture(ROOT), indent=2))

    train, val = load_all("train", limit, tasks, seed), load_all("val", limit, tasks, seed)
    set_seed(seed)  # BEFORE building the model: task-head initialisation must depend on the seed
    if build is not None:
        model, tok = build(cfg)
    else:
        tok = HFTokenizer(cfg["model_name"])
        model = MultiTaskTransformer.from_pretrained(cfg["model_name"], tasks=tuple(tasks), **cfg.get("model", {}))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="run_"))
    try:
        res = train_multitask(model, tok, train, val, tc, tmp, resume_path=out / "resume.pt")
        shutil.copy(tmp / "history.json", out / "history.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    train_s = time.time() - t0

    cal = _fit_calibration(model, tok, val, tc)
    t1 = time.time()
    val_m, val_p = _predict_split(model, tok, val, tc, cal)
    for task, df in val_p.items():
        df.to_csv(out / f"predictions_val_{task}.csv", index=False)
    infer_s = time.time() - t1
    metrics = {"experiment": cfg["name"], "seed": seed, "tasks": tasks, "model_name": cfg["model_name"],
               "n_parameters": model.n_parameters(), "best_epoch": res["best_epoch"], "epochs_run": len(res["history"]),
               "best_val_mean_macro_f1": res["best"], "train_seconds": round(train_s, 1),
               "val_inference_seconds": round(infer_s, 1), "device": device,
               "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
               "calibration": cal, "val": val_m, "test": None}
    if eval_test:
        test = load_all("test", None, tasks, seed)
        test_m, test_p = _predict_split(model, tok, test, tc, cal)
        for task, df in test_p.items():
            df.to_csv(out / f"predictions_test_{task}.csv", index=False)
        metrics["test"] = test_m
    if cfg.get("save_checkpoint"):
        save_checkpoint(out / "checkpoint", model, tok, {"train_config": asdict(tc), "name": cfg["name"], "best_epoch": res["best_epoch"]})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))  # written last = run complete
    (out / "resume.pt").unlink(missing_ok=True)
    print(f"done {cfg['name']} seed {seed}: train {train_s/60:.1f} min, best epoch {res['best_epoch']}, "
          f"val macro-F1 { {t: round(m['macro_f1'] or 0, 4) for t, m in val_m.items()} }", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
