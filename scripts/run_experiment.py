"""Run an experiment config for one or more seeds.

  python scripts/run_experiment.py configs/R2_muril_multitask_baseline.json
  python scripts/run_experiment.py configs/E1_3_lr_grid.json --seeds 13 21
Test-set evaluation happens only if the config says "evaluate_test": true (pre-registered final runs).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.runner import expand, run_one  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the config's seeds")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task/split (debug only; never for reported runs)")
    a = ap.parse_args()
    for path in a.configs:
        base = json.loads(Path(path).read_text())
        for cfg in expand(base):
            for seed in a.seeds or cfg["seeds"]:
                run_one(cfg, seed, a.device, limit=a.limit)
