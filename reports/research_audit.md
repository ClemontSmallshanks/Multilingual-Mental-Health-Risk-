# Research audit (Stage 1)

*Date: 2026-10-22. Scope: repository at commit `3588dd7`. No code, data split or model was changed for this audit.*

> **Follow-up status (2026-10-24):** A1 → 3 seeds (5 not feasible on CPU) · A2/A4 → not tuned (LR grid not run) · A3 → tested, head+tail did not help · A5 → fixed (per-label vs exact-set rates) · A6 → per-language/per-source reporting everywhere · A7 → open · A8 → done (single- vs multi-task ablation) · A9 → deletion test run (`results_summary.md` §7) · A10 → bootstrap now covers all tasks incl. emotion. Frozen splits, pinned packages, per-run environment capture and resumable training were added. See `results_summary.md`.

Everything below was checked directly against the code, the processed data, the checkpoints and the committed predictions. Where a statement is an inference rather than a measurement, it is labelled **(inference)**.

---

## 1. What currently works (verified)

| Item | How it was verified |
|---|---|
| Headline test metrics are internally consistent | Recomputed from `reports/metrics/predictions_*.csv` for all 3 models: sentiment macro-F1, risk F1, risk PR-AUC, risk ROC-AUC and emotion macro-F1 (masked) all match the JSON reports to 4 d.p. |
| Test set not used for model selection | Early stopping, temperature scaling and emotion thresholds all use the `val` split (`src/training/trainer.py`, `src/evaluation/evaluate.py`). |
| Leakage controls exist | Exact + near-duplicate removal with test > val > train priority; group splits by Dreaddit `post_id` and YouTube `video_id` (`src/data/prepare.py`, unit-tested). |
| Missing emotion labels handled | English BRIGHTER has no `disgust` annotation; stored as `-1`, masked in the loss and metrics (unit-tested). |
| Pipeline is end-to-end runnable | 59 unit tests pass; smoke runs for BiLSTM and transformer; API/UI render with real checkpoints. |
| Privacy logging | Test asserts a marker string never reaches logs across `/predict`, explain and batch. |

## 2. Incorrect, fragile or unverified

| # | Finding | Evidence | Severity |
|---|---|---|---|
| A1 | **Single seed for every model.** All comparisons rest on seed 13. Fine-tuning variance across seeds is known to be large for BERT-style models (Dodge et al., 2020; Mosbach et al., 2021). | `checkpoints/*/meta.json` | High |
| A2 | **Transformer selected at its last epoch (3/3) while validation loss rose.** Val loss went sentiment 0.955→1.004 and risk 0.524→0.616 from epoch 2→3, while macro-F1 rose slightly. The model is becoming over-confident; it may also be under-trained or mis-tuned. The epoch budget (3) was fixed by CPU time, not by validation. | `checkpoints/transformer/history.json` | High |
| A3 | **23.6 % of Dreaddit (stress) inputs are truncated** at `max_len=128` MuRIL tokens (median 103, p95 179). Information at the end of posts is discarded. Truncation elsewhere is ≤ 2.3 %. | Tokenised `data/processed/risk.csv` with the saved tokenizer | High (cheap to fix) |
| A4 | **Head LR is 16.7× the encoder LR** (heads `lr*0.5 = 5e-4`, encoder `3e-5`). Not necessarily wrong, but untested and undocumented as a choice. | `src/training/trainer.py:108` | Medium |
| A5 | **"Emotion error rate" is mislabelled.** `error_summary.json` / `error_analysis.csv` count an emotion example as an error if *any* annotated label differs (exact-set mismatch): 55.6 % for MuRIL. The per-label error rate is 16.4 %. Reports don't say which definition is used, so the number looks alarming and isn't comparable with F1. | Recomputed from predictions | Medium (reporting) |
| A6 | **Pooled metrics mix populations.** "Sentiment macro-F1" pools English tweets with two Hinglish sources; "emotion macro-F1" pools English and Hindi, which have different label distributions (English `fear` is positive in 58 % of training rows vs 15 % in Hindi) and different label sets (no English `disgust`). Pooled numbers should not be headline results. | `data/processed/stats.json` | Medium |
| A7 | **The same validation set does three jobs**: checkpoint selection, temperature fitting and emotion-threshold tuning. Calibration/threshold estimates are therefore slightly optimistic **(inference)**; no held-out calibration split. | Code review | Low-Medium |
| A8 | **Multi-task benefit never tested.** No single-task control exists, so nothing can be said about positive or negative transfer. | Code review | High (research gap) |
| A9 | **Explanations (Integrated Gradients) were never validated for faithfulness.** | Code review | Medium |
| A10 | **Bootstrap CIs cover sentiment and risk only**, not emotion, and resample test examples only (not seeds). | `scripts/bootstrap_ci.py` | Medium |

## 3. Data leakage and contamination risks

1. **Same-author leakage cannot be excluded.** No dataset ships author IDs. Dreaddit is grouped by post, YouTube by video, tweets not at all.
2. **"Hindi" tweets are Romanized and partly plain English.** The language-ID probe labels 26 % of the `tweets-hindi` test set as English (`reports/metrics/langid.json`). Any "Hinglish gap" estimate is contaminated by English examples labelled `hi-Latn`. This must be controlled before any claim about the language gap.
3. **Pre-trained checkpoint contamination (future risk).** Candidate checkpoints fine-tuned on Twitter sentiment are likely trained on overlapping data:
   * `cardiffnlp/twitter-roberta-base-sentiment-latest` is fine-tuned for sentiment on tweets (TimeLMs; Loureiro et al., 2022). The English UMSAB tweets were built from earlier SemEval data (Barbieri et al., 2022). **Overlap with our English test set must be checked before use; otherwise use only non-fine-tuned backbones.**
   * Any `*-sentiment` model trained on UMSAB itself (XLM-T) is directly contaminated and must not be used.
4. **Hinglish (YouTube) split rests on 23 videos.** Group split is correct but coarse; the test set is a few videos, so results can swing with the video draw.
5. **No cross-dataset contamination audit** between the four datasets (e.g. near-duplicate tweets across configs). Low risk **(inference)** but unchecked.

## 4. Metric inconsistencies

* A5 (emotion "error rate" definition) is the only true inconsistency found. All headline metrics reproduce exactly.
* ECE uses 15 equal-width bins (top-label); emotion ECE is computed over flattened label cells. This is reasonable, but it is not documented in the reports.
* The README's "single training run" caveat is accurate; it should be made prominent in any research write-up.

## 5. Reproducibility gaps

| Gap | Fix (Stage 2) |
|---|---|
| `requirements.txt` unpinned; no environment record | Pin versions; write `environment.json` per run (python, torch, transformers, sklearn, CPU/GPU) |
| No config files; settings live in argparse defaults | Add a `configs/*.yaml|json` per experiment; save resolved config with every run |
| Split membership not frozen as an artefact | Commit `splits/*.csv` (text_id → split, *no text*) plus a hash, and assert it at load time |
| No deterministic flags | `torch.use_deterministic_algorithms(True)` where supported; record that GPU kernels may still vary |
| Only one seed | ≥ 5 seeds (13, 21, 42, 55, 77) for final comparisons |
| Per-run outputs overwrite each other (`class_balance.json`, `comparison.json`) | Per-run directory `runs/<exp_id>/<seed>/` with metrics, predictions, logs |
| Training duration recorded only in logs | Store train/inference time and peak memory in run metadata |

## 6. Potential research contributions (to be earned, not assumed)

1. **Controlled multi-task vs single-task study** across three heterogeneous datasets/populations, with negative transfer reported per task.
2. **Decomposition of the English–Hinglish sentiment gap**: language vs source/domain vs label noise vs English contamination, plus tested interventions (Hinglish-specific encoder, transliteration).
3. **Reproducible, multi-seed re-evaluation** of a small Indic-capable stack on Dreaddit / UMSAB / BRIGHTER with calibration and selective prediction.
4. **Faithfulness check of Integrated-Gradients explanations** on English vs Hinglish text.

## 7. Risks that could invalidate final claims

* **Compute.** One MuRIL run takes ~2 h on the available 4-core CPU. 5 seeds × several configs is ~30–60 CPU-hours. Without a GPU, the scope must shrink (see `experiment_plan.md` §6).
* **Small, noisy Hinglish data.** 1,343 test examples from two sources; effects of 1–3 F1 points may not be resolvable.
* **Confounds.** Language, platform, annotator and label distribution all change together between English and Hinglish data. Without stratification, a "language gap" claim is not supported.
* **Test-set reuse.** Running many configurations and looking at test results would turn the test set into a dev set. All selection must happen on validation; test is evaluated once per pre-registered configuration.
* **Stress ≠ clinical risk.** Any write-up must keep the Dreaddit stress label separate from mental-health status.

## 8. Prioritised plan (summary; full detail in `experiment_plan.md`)

| Priority | Work | Why | Cost |
|---|---|---|---|
| P0 | Reproducibility fixes (§5), relabel emotion error metric (A5), frozen splits | Every later claim depends on it | Low |
| P1 | Multi-seed baseline reproduction (MuRIL, TF-IDF) | A1: current numbers are single-seed | Medium |
| P2 | **Major improvement 1 — stress detection:** fix truncation (A3) + LR/epoch tuning (A2, A4) | Cheapest high-expected-value change on the main task | Medium |
| P3 | **Major improvement 2 — Hinglish gap:** clean the `hi-Latn` set, then compare MuRIL vs Hinglish-specific encoder vs transliteration | Primary research question | Medium-High |
| P4 | Multi-task vs single-task ablation (A8) | Core methodological question | High |
| P5 | Calibration + selective prediction; IG faithfulness (A9) | Reliability story | Low-Medium |
