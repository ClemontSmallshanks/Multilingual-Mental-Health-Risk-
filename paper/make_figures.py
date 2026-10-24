"""Figures for the paper, generated from reports/results/*.json and runs/ (no hand-typed values except where a
number is copied from a verified published table, marked in the caption)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
C = {"muril": "#555555", "hing": "#111111", "light": "#BBBBBB"}
summary = pd.read_csv(ROOT / "reports/results/summary.csv")


def val(exp, scope, split="test", task="sentiment", metric="macro_f1"):
    r = summary[(summary.experiment == exp) & (summary.split == split) & (summary.task == task) & (summary.scope == scope) & (summary.metric == metric)].iloc[0]
    return float(r["mean"]), float(r["sd"])


# Fig. 2: sentiment macro-F1 by language/source (test), MuRIL vs HingRoBERTa, error bars = s.d. over 3 seeds
scopes = [("language:en-Latn", "English\ntweets"), ("source:tweets-hindi", "Romanized\nHindi tweets"), ("source:youtube-hinglish", "Hinglish\nYouTube")]
fig, ax = plt.subplots(figsize=(3.45, 2.2), dpi=300)
x = np.arange(len(scopes)); w = 0.36
for i, (exp, lab, col, hatch) in enumerate((("C_ST_sentiment_muril", "MuRIL", C["light"], ""), ("C_ST_sentiment_hingroberta", "HingRoBERTa", C["muril"], "//"))):
    m = [val(exp, s)[0] for s, _ in scopes]; sd = [val(exp, s)[1] for s, _ in scopes]
    ax.bar(x + (i - 0.5) * w, m, w, yerr=sd, capsize=2, color=col, edgecolor="black", linewidth=0.5, hatch=hatch, label=lab, error_kw={"linewidth": 0.6})
    for xi, mi in zip(x + (i - 0.5) * w, m):
        ax.text(xi, mi + 0.035, f"{mi:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in scopes], fontsize=7); ax.set_ylim(0.4, 0.8)
ax.set_ylabel("Test macro-F1"); ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig_sentiment_language.png"); plt.close(fig)

# Fig. 3: risk-coverage (seed-averaged, thresholds chosen on validation)
sp = json.load(open(ROOT / "reports/results/selective_prediction.json"))
fig, ax = plt.subplots(figsize=(3.45, 2.3), dpi=300)
spec = [("C_ST_risk_head128", "risk", "Stress (MuRIL)", "o", "-"), ("C_ST_sentiment_hingroberta", "sentiment", "Sentiment (HingRoBERTa)", "s", "--"), ("C_ST_sentiment_muril", "sentiment", "Sentiment (MuRIL)", "^", ":")]
for exp, task, lab, mk, ls in spec:
    c = sp[exp][task]["curve_mean"]; xs = [float(k) for k in c]; ys = [v["macro_f1"] for v in c.values()]
    ax.plot(xs, ys, marker=mk, ms=3, lw=1, ls=ls, color="black", label=lab, markerfacecolor="white" if mk != "o" else "black")
ax.invert_xaxis(); ax.set_xlabel("Coverage (fraction of test inputs answered)"); ax.set_ylabel("Macro-F1 on answered inputs")
ax.legend(frameon=False, fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25, linewidth=0.4)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig_risk_coverage.png"); plt.close(fig)

# Fig. 4: IG faithfulness (deletion test)
f = json.load(open(ROOT / "reports/results/faithfulness.json"))["groups"]
names = [("risk:en-Latn", "English\nstress"), ("sentiment:en-Latn", "English\nsentiment"), ("sentiment:hi-Latn", "Hinglish\nsentiment")]
fig, ax = plt.subplots(figsize=(3.45, 2.2), dpi=300)
x = np.arange(len(names)); w = 0.26
for i, (key, lab, col, hatch) in enumerate((("mean_drop_top_k", "top-k deleted", "black", ""), ("mean_drop_random_k", "random-k", C["muril"], "//"), ("mean_drop_bottom_k", "bottom-k", C["light"], ".."))):
    ax.bar(x + (i - 1) * w, [f[k][key] for k, _ in names], w, color=col, edgecolor="black", linewidth=0.5, hatch=hatch, label=lab)
ax.axhline(0, color="black", linewidth=0.5); ax.set_xticks(x); ax.set_xticklabels([l for _, l in names], fontsize=7)
ax.set_ylabel("Drop in predicted-class\nprobability"); ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig_faithfulness.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
