"""Aggregate experiment runs and compare experiments with a paired bootstrap.

* collect(): one row per (experiment, seed, split, task, scope, metric) from every metrics.json
* summarise(): mean, s.d., min, max, n_seeds per (experiment, split, task, scope, metric)
* paired_bootstrap(): difference B - A of a seed-averaged metric, resampling *examples* (same resample for both
  experiments and all seeds; Berg-Kirkpatrick et al., 2012). Seed variance is reported separately via summarise();
  the bootstrap CI alone does not capture it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import EMOTION_LABELS, TASK_LABELS

METRICS = ("macro_f1", "weighted_f1", "accuracy", "f1", "precision", "recall", "pr_auc", "roc_auc", "ece", "brier",
           "per_label_error_rate", "exact_set_mismatch_rate")


def _rows(exp, seed, split, task, scope, m):
    return [{"experiment": exp, "seed": seed, "split": split, "task": task, "scope": scope, "metric": k, "value": m[k]}
            for k in METRICS if isinstance(m.get(k), (int, float))]


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(runs_dir).glob("*/seed_*/metrics.json")):
        m = json.loads(f.read_text())
        for split in ("val", "test"):
            for task, tm in (m.get(split) or {}).items():
                rows += _rows(m["experiment"], m["seed"], split, task, "all", tm)
                for kind in ("by_language", "by_source"):
                    for g, gm in tm.get(kind, {}).items():
                        rows += _rows(m["experiment"], m["seed"], split, task, f"{kind[3:]}:{g}", gm)
        rows.append({"experiment": m["experiment"], "seed": m["seed"], "split": "run", "task": "-", "scope": "-",
                     "metric": "train_minutes", "value": m["train_seconds"] / 60})
        rows.append({"experiment": m["experiment"], "seed": m["seed"], "split": "run", "task": "-", "scope": "-",
                     "metric": "best_epoch", "value": m["best_epoch"]})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["experiment", "split", "task", "scope", "metric"]).value
    return g.agg(mean="mean", sd="std", min="min", max="max", n_seeds="count").reset_index()


# ---- paired bootstrap on per-example predictions -------------------------------------------------------
def _metric(task, metric, d: pd.DataFrame, thr=None) -> float:
    if task == "emotion":
        fs = []
        for j, e in enumerate(EMOTION_LABELS):
            y = d[f"y_{e}"].to_numpy(); m = y >= 0
            if m.sum() and y[m].sum():
                fs.append(f1_score(y[m], (d[f"p_{e}"].to_numpy()[m] >= thr[j]).astype(int), zero_division=0))
        return float(np.mean(fs)) if fs else np.nan
    labels = TASK_LABELS[task]
    P = d[[f"p_{l}" for l in labels]].to_numpy()
    y = d["y"].to_numpy()
    if metric == "macro_f1":
        return f1_score(y, P.argmax(1), average="macro", labels=list(range(len(labels))), zero_division=0)
    if metric == "f1":
        return f1_score(y, P.argmax(1), zero_division=0)
    if len(set(y)) < 2:
        return np.nan
    if metric == "pr_auc":
        return average_precision_score(y, P[:, 1])
    if metric == "roc_auc":
        return roc_auc_score(y, P[:, 1])
    raise ValueError(metric)


def _load(runs_dir: Path, exp: str, split: str, task: str, scope: str | None):
    out = []
    for sd in sorted((Path(runs_dir) / exp).glob("seed_*")):
        f = sd / f"predictions_{split}_{task}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if scope:
            kind, _, val = scope.partition(":")
            d = d[d[kind] == val]
        thr = json.loads((sd / "metrics.json").read_text())["calibration"]["emotion_thresholds"]
        out.append((int(sd.name.split("_")[1]), d.sort_values("text_id").reset_index(drop=True), thr))
    return out


def paired_bootstrap(runs_dir, exp_a, exp_b, split, task, metric, scope=None, n=1000, seed=0) -> dict:
    A, B = _load(runs_dir, exp_a, split, task, scope), _load(runs_dir, exp_b, split, task, scope)
    if not A or not B:
        raise ValueError(f"missing predictions for {exp_a if not A else exp_b} ({split})")
    ids = A[0][1].text_id.tolist()
    for _, d, _ in A + B:
        if d.text_id.tolist() != ids:
            raise ValueError("experiments were evaluated on different examples; refusing to compare")
    rng = np.random.default_rng(seed)
    point_a = np.mean([_metric(task, metric, d, t) for _, d, t in A])
    point_b = np.mean([_metric(task, metric, d, t) for _, d, t in B])
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(ids), len(ids))
        ma = np.nanmean([_metric(task, metric, d.iloc[i], t) for _, d, t in A])
        mb = np.nanmean([_metric(task, metric, d.iloc[i], t) for _, d, t in B])
        diffs.append(mb - ma)
    diffs = np.array(diffs)
    valid = diffs[~np.isnan(diffs)]  # resamples where the metric is undefined (e.g. one class only) are dropped
    lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
    return {"a": exp_a, "b": exp_b, "split": split, "task": task, "scope": scope or "all", "metric": metric,
            "seeds_a": [s for s, _, _ in A], "seeds_b": [s for s, _, _ in B], "n_examples": len(ids),
            "mean_a": float(point_a), "mean_b": float(point_b), "diff_b_minus_a": float(point_b - point_a),
            "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_resamples": n, "n_valid_resamples": int(len(valid))}
