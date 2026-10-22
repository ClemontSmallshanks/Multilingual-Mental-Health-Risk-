# Reproducibility

*Status (2026-10-22): the framework below is implemented and tested (66 unit tests; end-to-end dry run with the real MuRIL and HingRoBERTa checkpoints on a 24-example subset). **No Stage 2/3 experiment has been run yet.** Existing results in `reports/metrics/` are the original single-seed CPU runs.*

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
