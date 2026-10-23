"""Evaluate an already-trained, validation-selected experiment on the TEST split, once.

Used only for configurations chosen on validation (experiment_plan.md Sec. 7, "best of ... on val").
Loads each seed's saved checkpoint (no retraining), applies the calibration fitted on validation during the run,
writes predictions_test_<task>.csv and adds "test" + "test_evaluated_after_selection" to metrics.json.
Refuses to run twice.

  python scripts/evaluate_selected_on_test.py C_ST_risk_headtail256 --reason "beat C_ST_risk_head128 on val PR-AUC"
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.datasets import load_all  # noqa: E402
from src.experiments.runner import _predict_split, runs_dir  # noqa: E402
from src.models.io import load_checkpoint  # noqa: E402
from src.training.trainer import TrainConfig  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--reason", required=True, help="the validation evidence that selected this config")
    a = ap.parse_args()
    for sd in sorted((runs_dir() / a.experiment).glob("seed_*")):
        mf = sd / "metrics.json"
        m = json.loads(mf.read_text())
        if m.get("test") is not None:
            sys.exit(f"{sd} already has test metrics; refusing to evaluate the test set twice.")
        cfg = json.loads((sd / "config.json").read_text())
        tc = TrainConfig(**{k: v for k, v in cfg["resolved_train_config"].items()})
        tc.device = "cpu"
        model, tok, _ = load_checkpoint(sd / "checkpoint", "cpu")
        test = load_all("test", None, m["tasks"], m["seed"])
        tm, tp = _predict_split(model, tok, test, tc, m["calibration"])
        for task, df in tp.items():
            df.to_csv(sd / f"predictions_test_{task}.csv", index=False)
        m["test"] = tm
        m["test_evaluated_after_selection"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": a.reason}
        mf.write_text(json.dumps(m, indent=2, default=float))
        print(f"{sd.name}: test evaluated", {t: round(v.get('macro_f1') or 0, 4) for t, v in tm.items()})
