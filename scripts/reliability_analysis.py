"""Selective prediction / reliability analysis from saved run predictions (experiment_plan.md E4.1-E4.2).

For each experiment that has TEST predictions (sentiment / risk; single-label tasks):
  * risk-coverage curve on test: macro-F1 and accuracy when keeping only the most confident x% (confidence = max prob)
  * AURC (area under the error-rate/coverage curve; lower is better; Geifman & El-Yaniv 2017)
  * an abstention threshold chosen on VALIDATION to keep 80 % of examples, then applied unchanged to test
Results are seed-averaged. Writes reports/results/selective_prediction.json and reports/figures/risk_coverage.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FIGURES_DIR, REPORTS_DIR, TASK_LABELS  # noqa: E402
from src.experiments.runner import runs_dir  # noqa: E402

COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def load(path, task):
    d = pd.read_csv(path)
    P = d[[f"p_{l}" for l in TASK_LABELS[task]]].to_numpy()
    return d, P, d["y"].to_numpy()


def curve(P, y):
    conf, pred = P.max(1), P.argmax(1)
    order = np.argsort(-conf)
    out = {}
    for c in COVERAGES:
        k = max(1, int(round(c * len(y))))
        idx = order[:k]
        out[str(c)] = {"macro_f1": float(f1_score(y[idx], pred[idx], average="macro")), "accuracy": float((pred[idx] == y[idx]).mean())}
    err = (pred[order] != y[order]).astype(float)
    aurc = float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))
    return out, aurc


if __name__ == "__main__":
    res, plots = {}, {}
    for exp_dir in sorted(runs_dir().glob("*")):
        for task in ("sentiment", "risk"):
            seeds = sorted(exp_dir.glob(f"seed_*/predictions_test_{task}.csv"))
            if not seeds:
                continue
            per_seed = []
            for f in seeds:
                _, Pt, yt = load(f, task)
                _, Pv, yv = load(f.parent / f"predictions_val_{task}.csv", task)
                thr = float(np.quantile(Pv.max(1), 0.2))  # keep 80 % on validation
                keep = Pt.max(1) >= thr
                c, aurc = curve(Pt, yt)
                pred = Pt.argmax(1)
                per_seed.append({"curve": c, "aurc": aurc, "val_threshold": thr, "test_coverage": float(keep.mean()),
                                 "test_macro_f1_kept": float(f1_score(yt[keep], pred[keep], average="macro")),
                                 "test_macro_f1_all": float(f1_score(yt, pred, average="macro"))})
            agg = {"seeds": len(per_seed), "aurc_mean": float(np.mean([s["aurc"] for s in per_seed])),
                   "curve_mean": {c: {m: float(np.mean([s["curve"][c][m] for s in per_seed])) for m in ("macro_f1", "accuracy")} for c in map(str, COVERAGES)},
                   "abstain_at_val80": {k: float(np.mean([s[k] for s in per_seed])) for k in ("test_coverage", "test_macro_f1_kept", "test_macro_f1_all")},
                   "per_seed": per_seed}
            res.setdefault(exp_dir.name, {})[task] = agg
            plots[f"{exp_dir.name}:{task}"] = agg["curve_mean"]
    out = REPORTS_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selective_prediction.json").write_text(json.dumps(res, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    for name, c in plots.items():
        ax.plot([float(k) for k in c], [v["macro_f1"] for v in c.values()], marker="o", ms=3, label=name)
    ax.set_xlabel("coverage (fraction of test examples answered)"); ax.set_ylabel("macro-F1 on answered examples")
    ax.invert_xaxis(); ax.grid(alpha=0.2); ax.legend(fontsize=6, frameon=False)
    ax.set_title("Risk-coverage (confidence = max calibrated probability), seed-averaged", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "risk_coverage.png"); plt.close(fig)
    for e, t in res.items():
        for task, a in t.items():
            print(f"{e:30s} {task:9s} AURC {a['aurc_mean']:.3f}  F1@100% {a['curve_mean']['1.0']['macro_f1']:.3f}  F1@80% {a['curve_mean']['0.8']['macro_f1']:.3f}  "
                  f"val-chosen thr -> test coverage {a['abstain_at_val80']['test_coverage']:.2f}, F1 kept {a['abstain_at_val80']['test_macro_f1_kept']:.3f}")
