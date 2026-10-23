"""Stage 2 experiment framework: truncation, frozen splits, runner, aggregation. Offline, tiny models, synthetic fixtures."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data import datasets
from src.data.splits import SplitMismatchError, freeze, verify
from src.experiments.aggregate import collect, paired_bootstrap, summarise
from src.experiments.runner import expand, run_one


def test_head_tail_truncation_keeps_start_and_end(tiny_transformer):
    _, tok = tiny_transformer
    text = " ".join(["the"] * 3 + ["weather"] * 40 + ["nice"] * 3)
    head = tok.encode([text], 12)["input_ids"][0].tolist()
    ht = tok.encode([text], 12, truncation="head_tail", head_tokens=3)["input_ids"][0].tolist()
    ids = lambda w: tok.tok.convert_tokens_to_ids(w)
    assert len(ht) == 12 and ht[0] == ids("[CLS]") and ht[-1] == ids("[SEP]")
    assert ht[1:4] == [ids("the")] * 3 and ht[-4:-1] == [ids("nice")] * 3  # tail kept
    assert ids("nice") not in head  # plain head truncation loses the end
    short = tok.encode(["the weather"], 12, truncation="head_tail", head_tokens=3)
    assert short["input_ids"].tolist() == tok.encode(["the weather"], 12)["input_ids"].tolist()  # no-op when it fits
    with pytest.raises(ValueError):
        tok.encode([text], 12, truncation="middle")


def test_split_freeze_and_verify_detects_drift(processed_dir, tmp_path):
    sd = tmp_path / "splits"
    freeze(processed_dir, sd)
    df = pd.read_csv(processed_dir / "risk.csv", keep_default_na=False)
    assert verify("risk", df, sd) is True
    moved = df.copy(); moved.loc[0, "split"] = "test"
    with pytest.raises(SplitMismatchError):
        verify("risk", moved, sd)
    edited = df.copy(); edited.loc[1, "text"] = edited.loc[1, "text"] + " changed"
    with pytest.raises(SplitMismatchError):
        verify("risk", edited, sd)
    assert verify("risk", df, tmp_path / "absent") is False
    assert "text" not in pd.read_csv(sd / "risk.csv").columns  # no raw text stored


def test_load_task_refuses_drifted_data(processed_dir, tmp_path, monkeypatch):
    sd = tmp_path / "splits2"
    freeze(processed_dir, sd)
    df = pd.read_csv(processed_dir / "risk.csv", keep_default_na=False)
    df.loc[0, "split"] = "val"
    df.to_csv(processed_dir / "risk.csv", index=False)
    monkeypatch.setattr("src.data.datasets.SPLITS_DIR", sd)
    with pytest.raises(SplitMismatchError):
        datasets.load_task("risk", "train")


def test_grid_expansion():
    c = {"name": "g", "train": {"epochs": 2}, "grid": {"encoder_lr": [1, 2], "head_lr": [3, 4]}}
    out = expand(c)
    assert len(out) == 4 and len({x["name"] for x in out}) == 4 and all("grid" not in x for x in out)
    assert out[0]["train"]["epochs"] == 2 and expand({"name": "x"}) == [{"name": "x"}]


def _cfg(name, tasks, test=False, **train):
    return {"name": name, "model_name": "tiny-test", "tasks": tasks, "seeds": [1], "evaluate_test": test,
            "train": {"epochs": 1, "batch_size": 8, "patience": 1, **train}}


def test_runner_end_to_end_and_test_discipline(processed_dir, tmp_path, monkeypatch, tiny_transformer):
    monkeypatch.setenv("MHRISK_RUNS_DIR", str(tmp_path / "runs"))
    build = lambda cfg: tiny_transformer
    out = run_one(_cfg("val_only", ["sentiment", "emotion", "risk"], head_lr=1e-3), 1, "cpu", build=build)
    m = json.loads((out / "metrics.json").read_text())
    assert m["test"] is None and not list(out.glob("predictions_test*"))  # test untouched unless pre-registered
    assert set(m["val"]) == {"sentiment", "emotion", "risk"} and m["best_epoch"] == 1
    assert {"per_label_error_rate", "exact_set_mismatch_rate"} <= set(m["val"]["emotion"])
    cfgj = json.loads((out / "config.json").read_text())
    assert cfgj["resolved_train_config"]["head_lr"] == 1e-3 and json.loads((out / "env.json").read_text())["python"]
    cols = set().union(*(pd.read_csv(out / f"predictions_val_{t}.csv").columns for t in ("sentiment", "emotion", "risk")))
    assert "text" not in cols and {"p_stress", "p_negative", "y_anger", "p_disgust"} <= cols
    # resume: second call is a no-op
    mtime = (out / "metrics.json").stat().st_mtime
    run_one(_cfg("val_only", ["sentiment", "emotion", "risk"]), 1, "cpu", build=build)
    assert (out / "metrics.json").stat().st_mtime == mtime


def test_single_task_and_aggregate_and_bootstrap(processed_dir, tmp_path, monkeypatch, tiny_transformer, tiny_bilstm):
    from src.models.multitask import MultiTaskTransformer
    from transformers import BertConfig, BertModel

    runs = tmp_path / "runs"
    monkeypatch.setenv("MHRISK_RUNS_DIR", str(runs))
    tok = tiny_transformer[1]

    def build(cfg):
        enc = BertModel(BertConfig(vocab_size=len(tok.tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64))
        return MultiTaskTransformer(enc, "tiny", shared_dim=16, tasks=tuple(cfg["tasks"])), tok

    for name in ("A", "B"):
        for seed in (1, 2):
            run_one(_cfg(name, ["risk"], test=True), seed, "cpu", build=build)
    m = json.loads((runs / "A" / "seed_1" / "metrics.json").read_text())
    assert set(m["val"]) == {"risk"} and set(m["test"]) == {"risk"}
    df = collect(runs)
    s = summarise(df)
    row = s[(s.experiment == "A") & (s.split == "test") & (s.task == "risk") & (s.scope == "all") & (s.metric == "pr_auc")]
    assert len(row) == 1 and row.n_seeds.iloc[0] == 2
    r = paired_bootstrap(runs, "A", "B", "test", "risk", "pr_auc", n=50)
    assert r["seeds_a"] == [1, 2] and len(r["ci95"]) == 2 and r["n_valid_resamples"] > 0 and r["ci95"][0] <= r["ci95"][1]
    assert np.isclose(paired_bootstrap(runs, "A", "A", "test", "risk", "macro_f1", n=20)["diff_b_minus_a"], 0.0)


def test_emotion_bootstrap_reads_masked_labels(processed_dir, tmp_path, monkeypatch, tiny_transformer):
    runs = tmp_path / "runs"
    monkeypatch.setenv("MHRISK_RUNS_DIR", str(runs))
    run_one(_cfg("E", ["sentiment", "emotion", "risk"]), 1, "cpu", build=lambda c: tiny_transformer)
    d = pd.read_csv(runs / "E" / "seed_1" / "predictions_val_emotion.csv")
    assert d.y_disgust.dtype.kind == "i" and (d.y_disgust == -1).any()  # masked labels survive the round trip
    r = paired_bootstrap(runs, "E", "E", "val", "emotion", "macro_f1", n=10)
    assert r["diff_b_minus_a"] == 0.0 and r["n_valid_resamples"] > 0


def test_training_resumes_after_interruption(processed_dir, tmp_path, tiny_transformer, monkeypatch):
    """Simulate a machine restart mid-epoch: the second call must pick up from the snapshot and finish."""
    import src.training.trainer as tr
    from src.data.datasets import load_all

    model, tok = tiny_transformer
    cfg = tr.TrainConfig(kind="transformer", epochs=2, batch_size=4, patience=5, save_checkpoint=False, resume_every_s=0)
    resume = tmp_path / "resume.pt"
    calls = {"n": 0}
    real = tr.task_loss

    def crashing_loss(*a, **k):
        calls["n"] += 1
        if calls["n"] == 7:  # crash in the middle of epoch 1
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(tr, "task_loss", crashing_loss)
    with pytest.raises(KeyboardInterrupt):
        tr.train_multitask(model, tok, load_all("train"), load_all("val"), cfg, tmp_path / "o", resume_path=resume)
    snap = __import__("torch").load(resume, weights_only=False)
    assert snap["epoch"] == 1 and snap["batch_idx"] == 6 and snap["step"] == 6
    monkeypatch.setattr(tr, "task_loss", real)
    res = tr.train_multitask(model, tok, load_all("train"), load_all("val"), cfg, tmp_path / "o", resume_path=resume)
    n_batches = sum(-(-len(d) // 4) for d in load_all("train").values())
    assert len(res["history"]) == 2 and res["history"][0]["resumed_within_epoch"] is True
    final = __import__("torch").load(resume, weights_only=False)
    assert final["finished"] and final["step"] == 2 * n_batches  # no batch skipped or repeated
    again = tr.train_multitask(model, tok, load_all("train"), load_all("val"), cfg, tmp_path / "o", resume_path=resume)
    assert len(again["history"]) == 2  # already finished: nothing retrained


def test_same_seed_gives_identical_head_init(processed_dir, tmp_path, monkeypatch, tiny_transformer):
    """Regression: heads used to be initialised before seeding, so 'seed 13' differed between processes."""
    import torch
    from src.experiments import runner
    from src.models.multitask import MultiTaskTransformer
    from transformers import BertConfig, BertModel

    tok = tiny_transformer[1]
    inits = []

    def build(cfg):
        enc = BertModel(BertConfig(vocab_size=len(tok.tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64))
        m = MultiTaskTransformer(enc, "tiny", shared_dim=16, tasks=("risk",))
        inits.append(m.heads["risk"].weight.detach().clone())
        return m, tok

    for i in range(2):
        torch.manual_seed(1000 + i)  # simulate different global RNG state in different processes
        monkeypatch.setenv("MHRISK_RUNS_DIR", str(tmp_path / f"r{i}"))
        runner.run_one(_cfg("S", ["risk"]), 7, "cpu", build=build)
    assert torch.equal(inits[0], inits[1])
