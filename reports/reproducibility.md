# Reproducibility

*Status (2026-10-24): all reported experiments were run with this framework on a 4-core CPU (fp32), configs in `configs/cpu/`, artefacts in `runs/` (committed: config, env, history, metrics, per-example probabilities; no text, no checkpoints). Results: `results_summary.md`. `reports/metrics/` still holds the original single-seed October runs.*

## Reproducing the reported results

```bash
python scripts/download_data.py && python scripts/prepare_data.py && python scripts/freeze_splits.py --check
scripts/run_cpu_queue.sh                                   # resumable; ~24 CPU-hours on 4 cores
python scripts/evaluate_selected_on_test.py C_ST_sentiment_hingroberta --reason "<validation evidence>"
python scripts/write_results_tables.py                     # tables + paired bootstrap -> reports/results/
python scripts/reliability_analysis.py                     # selective prediction
python scripts/faithfulness_test.py                        # IG deletion test
python -m pytest -q
```

Verified: multi-task seed 13 reproduces the original October run exactly (identical validation metrics per epoch).

## Robustness to machine restarts

Training saves a full snapshot (`runs/<exp>/seed_<n>/resume.pt`: weights, optimizer, scheduler, RNG states, position within the epoch, best weights) every 10 minutes and at every epoch end, written atomically. A restarted run resumes from it, so no batch is skipped or repeated (unit-tested). The snapshot is deleted when the run completes and is git-ignored. Caveats: dropout randomness after a resume differs from an uninterrupted run, and `train_minutes` in `metrics.json` counts only the final process.

## Fixed during the experiments

* Task heads were initialised *before* the seed was set in the experiment runner, so "seed 13" differed between processes. Fixed (seed set before model construction) and regression-tested; the affected partial run was discarded and restarted.

## What makes a result reproducible here

| Control | Where |
|---|---|
| Frozen split membership: `splits/<task>.csv` (text_id, split, SHA-1 of text; **no text**) + `splits/MANIFEST.json` (SHA-256) | `src/data/splits.py`; checked automatically on every data load, and by `python scripts/freeze_splits.py --check` |
| Pinned data-processing and modelling packages | `requirements-colab.txt` (GPU; torch from Colab), `requirements-lock.txt` (local CPU, Python 3.13) |
| One config file per experiment, resolved config saved per run | `configs/*.json`, `runs/<exp>/seed_<n>/config.json` |
| Seeds | final comparisons: 13, 21, 42, 55, 77 |
| Environment per run (python, torch, transformers, sklearn, GPU, git commit, dirty flag) | `runs/.../env.json` |
| Per-epoch train/val losses and val metrics | `runs/.../history.json` |
| Val (always) and test (pre-registered only) metrics, calibration, train time, peak GPU memory | `runs/.../metrics.json` (written last: its presence = run complete; reruns skip it) |
| Per-example calibrated probabilities (no text) | `runs/.../predictions_<split>_<task>.csv` |
| Test-set discipline | test predictions are only produced for configs with `"evaluate_test": true`: R2 and E3.1 (pre-registered in `experiment_plan.md` §7). Unit-tested. |

Determinism: `TrainConfig.deterministic=True` enables `torch.use_deterministic_algorithms(warn_only=True)`. GPU fp16 runs (`amp: true`) are **not** bit-identical across GPU types; seed-to-seed variation is reported instead.

## Commands

```bash
# data (raw text never committed)
python scripts/download_data.py && python scripts/prepare_data.py
python scripts/freeze_splits.py --check          # must print "matches frozen split" x3

# experiments (GPU recommended; see notebooks/colab_experiments.ipynb)
python scripts/run_experiment.py configs/R2_muril_multitask_baseline.json
python scripts/run_experiment.py configs/E1_3_lr_grid.json --seeds 13
export MHRISK_RUNS_DIR=/path/to/runs            # optional; default ./runs

# summaries and paired bootstrap (Berg-Kirkpatrick et al., 2012)
python scripts/aggregate_runs.py
python scripts/aggregate_runs.py --compare R2_muril_multitask_baseline E1_2_risk_headtail256 --split val --task risk --metric pr_auc
python scripts/aggregate_runs.py --compare E3_1_muril_single_sentiment E2_1_hingroberta_sentiment --split val --task sentiment --metric macro_f1 --scope language:hi-Latn
```

## Known differences from the original (seed-13) runs

* Original runs: CPU, fp32. New runs: GPU, fp16 autocast. R2 re-establishes the baseline under the new setting rather than reusing the CPU number.
* `head_lr` is now explicit. R2 sets it to 5e-4, which equals the legacy `lr * 0.5`, so R2 is the same configuration.
* Emotion error reporting now includes both `per_label_error_rate` and `exact_set_mismatch_rate` (audit A5).
