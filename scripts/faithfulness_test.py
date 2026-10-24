"""Deletion-based faithfulness test of the Integrated-Gradients explanations served by the app (plan E4.4).

For each validation example: compute IG attributions for the predicted class, then delete k = ceil(20 %) of the
(non-special) tokens and measure the drop in that class's probability (ERASER "comprehensiveness", DeYoung et al. 2020):
  top-k (highest attribution)  vs  random-k (mean of 5 draws)  vs  bottom-k (lowest attribution).
If IG is faithful, deleting top-k should hurt much more than random-k. Validation data only; texts are never written out.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MAX_LEN, REPORTS_DIR  # noqa: E402
from src.data.datasets import load_task  # noqa: E402
from src.models.io import load_checkpoint  # noqa: E402


def probs(model, ids, task):
    with torch.no_grad():
        return torch.softmax(model(ids, torch.ones_like(ids), task), -1)[0]


def ig(model, ids, task, target, steps=16):
    emb = model.embedding_layer()(ids).detach()
    mask = torch.ones_like(ids)
    total = torch.zeros_like(emb)
    for k in range(1, steps + 1):
        x = (k / steps * emb).requires_grad_(True)
        model.forward_from_embeddings(x, mask, task)[0, target].backward()
        total += x.grad
    return (emb * total / steps).sum(-1)[0].detach()


def delete(ids, positions):
    keep = [i for i in range(ids.shape[1]) if i not in set(positions)]
    return ids[:, keep]


def run(model, tok, task, texts, rng):
    rows = []
    for t in texts:
        ids = tok.encode([t], MAX_LEN[task])["input_ids"]
        content = list(range(1, ids.shape[1] - 1))  # skip [CLS]/[SEP]
        if len(content) < 3:
            continue
        p0 = probs(model, ids, task)
        target = int(p0.argmax())
        a = ig(model, ids, task, target)
        k = max(1, math.ceil(0.2 * len(content)))
        order = sorted(content, key=lambda i: -float(a[i]))
        drop = lambda pos: float(p0[target] - probs(model, delete(ids, pos), task)[target])
        rows.append({"top": drop(order[:k]), "bottom": drop(order[-k:]),
                     "random": float(np.mean([drop(rng.sample(content, k)) for _ in range(5)])), "n_tokens": len(content), "k": k})
    return rows


def summarise(rows, rng_np):
    top, rnd, bot = (np.array([r[x] for r in rows]) for x in ("top", "random", "bottom"))
    diffs = top - rnd
    boots = [diffs[rng_np.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
    return {"n": len(rows), "mean_drop_top_k": float(top.mean()), "mean_drop_random_k": float(rnd.mean()), "mean_drop_bottom_k": float(bot.mean()),
            "top_minus_random": float(diffs.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "share_examples_top_gt_random": float((diffs > 0).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/transformer")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng, rng_np = random.Random(0), np.random.default_rng(0)
    model, tok, meta = load_checkpoint(a.checkpoint, "cpu")
    out = {"checkpoint": a.checkpoint, "method": "IG (16 steps, zero baseline) on input embeddings; deletion of k=ceil(20%) content tokens", "groups": {}}
    groups = [("risk", "en-Latn"), ("sentiment", "en-Latn"), ("sentiment", "hi-Latn")]
    for task, lang in groups:
        d = load_task(task, "val").df
        d = d[d.language == lang].sample(n=a.n, random_state=0)
        rows = run(model, tok, task, d.text.tolist(), rng)
        out["groups"][f"{task}:{lang}"] = summarise(rows, rng_np)
        print(task, lang, json.dumps(out["groups"][f"{task}:{lang}"]), flush=True)
    p = REPORTS_DIR / "results" / "faithfulness.json"
    p.write_text(json.dumps(out, indent=2))
