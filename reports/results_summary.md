# Results summary (Stage 2–5)

*Date: 2026-10-24. Every number here comes from `runs/` via `scripts/write_results_tables.py` (`reports/results/results_tables.md`, `comparisons.json`) and the analysis scripts named below. Nothing is hand-entered from memory.*

## 0. What was actually run (and what was not)

**Compute.** No GPU was available: Kaggle accepted GPU requests from the API but attached only CPU machines without internet, even after phone verification, and Colab cannot be scripted. All experiments ran on a 4-core CPU in fp32, with mid-epoch resumable training because the machine was restarted repeatedly (each restart cost at most ~10 minutes).

**Protocol actually used (compute-reduced; deviations from `experiment_plan.md` in bold):**

| | Plan | Actual |
|---|---|---|
| Seeds for final comparisons | 5 | **3 (13, 21, 42)** |
| Stress experiments (M1) | multi-task | **single-task stress models** (≈¼ of the cost) + multi-task baseline |
| Faithfulness test (E4.4) | English vs Hinglish | as planned, comprehensiveness only (no sufficiency), 60 examples per group |
| LR grid (E1.3) | 9 cells × 10 epochs | **not run** (≈60 CPU-hours) |
| XLM-R control (E2.2) | 2 seeds | **not run** |
| Transliteration / normalisation (E2.3, E2.4) | val comparison | **not implemented** |
| Manual error taxonomy (§4.3) | 200 errors, 2 passes | **sheet exported, annotation not done** (needs a human annotator) |
| Everything else | as planned | as planned |

Test-set discipline was kept: test predictions exist only for the pre-registered runs (multi-task MuRIL, the three single-task MuRIL runs) and for HingRoBERTa, which was evaluated on test **once, after** it won on validation (selection evidence recorded in its `metrics.json`).

## 1. Headline results (test, mean ± s.d. over 3 seeds)

| Model | Sentiment macro-F1 | Emotion macro-F1 | Stress F1 | Stress PR-AUC | Stress ROC-AUC | Stress ECE |
|---|---|---|---|---|---|---|
| Multi-task MuRIL (baseline, original configuration) | 0.603 ± 0.008 | 0.684 ± 0.011 | 0.796 ± 0.014 | 0.852 ± 0.029 | 0.854 ± 0.017 | 0.035 ± 0.002 |
| Single-task MuRIL (one model per task) | 0.606 ± 0.003 | 0.682 ± 0.002 | 0.799 ± 0.005 | **0.875 ± 0.010** | 0.867 ± 0.005 | 0.062 ± 0.023 |
| Single-task **HingRoBERTa** (sentiment) | **0.648 ± 0.013** | – | – | – | – | – |

Reproducibility check: multi-task seed 13 reproduces the original October single-seed run **exactly** (identical validation metrics at every epoch).

## 2. Major improvement 1 — Hinglish sentiment (M2): **confirmed**

Hinglish-specific pre-training (HingRoBERTa; Nayak & Joshi, 2022) vs MuRIL, both single-task, same splits / hyper-parameters / seeds. Paired bootstrap (1,000 resamples of test examples; seed-averaged macro-F1):

| Test scope | MuRIL | HingRoBERTa | Δ [95 % CI] |
|---|---|---|---|
| All sentiment (n = 2,213) | 0.606 | 0.647 | **+0.042 [+0.025, +0.058]** |
| Hinglish / Romanized Hindi `hi-Latn` (n = 1,343) | 0.556 | 0.611 | **+0.056 [+0.032, +0.079]** |
| — tweets-hindi (n = 866) | 0.561 | 0.616 | **+0.054 [+0.028, +0.083]** |
| — youtube-hinglish (n = 477) | 0.531 | 0.602 | **+0.071 [+0.026, +0.114]** |
| English (n = 870) | 0.684 | 0.704 | +0.020 [−0.002, +0.044] (no loss) |

* The English–Hinglish gap shrinks from **0.128 → 0.093** (−27 %).
* All three classes improve (per-class F1 negative 0.643 → 0.675, neutral 0.524 → 0.578, positive 0.651 → 0.689).
* **Contamination check:** HingRoBERTa was pre-trained on Twitter text. Scanning all 52,933,356 lines of HingCorpus (third-party re-upload; line count matches the paper) found 1 exact (normalised) match among 1,824 Hinglish val/test texts, and **0 in test** (`reports/results/hingcorpus_overlap.json`). Near-duplicates were not checked.
* **Caveat — validation anomaly:** on the tweets-hindi *validation* split every model is near chance (≈0.35 macro-F1 vs ≈0.56–0.62 on its test split; TF-IDF shows the same pattern, English tweets do not). Selection was therefore effectively driven by the youtube-hinglish validation subset (+0.140 [+0.106, +0.176]); tweets-hindi validation showed no difference (−0.008 [−0.049, +0.037]). The test set then confirmed gains on *both* sources. Cause not verified (`reports/results/tweets_hindi_validation_anomaly.json`).
* Calibration cost: HingRoBERTa's sentiment ECE is 0.063 ± 0.010 vs 0.053 ± 0.011 for MuRIL (Brier improves 0.521 → 0.485).

## 3. Major improvement 2 — stress detection (M1): truncation hypothesis **rejected**; single-task model **improves ranking**

**H1a (truncation) — negative result.** 23.6 % of stress posts are truncated at 128 tokens, but head+tail truncation to 256 tokens (Sun et al., 2019) did **not** help on validation (selection only):

| Validation metric | 128 head | 256 head+tail | Δ [95 % CI] |
|---|---|---|---|
| macro-F1 | 0.792 | 0.772 | −0.020 [−0.037, −0.003] |
| PR-AUC | 0.883 | 0.876 | −0.006 [−0.017, +0.005] |
| ROC-AUC | 0.868 | 0.859 | −0.009 [−0.019, +0.001] |

Head truncation to 256 tokens (E1.1, separates "longer input" from "head+tail"): macro-F1 0.785 vs 0.792, Δ −0.007 [−0.024, +0.010]; PR-AUC 0.883 vs 0.883, Δ −0.000 [−0.011, +0.011]. **Longer inputs do not help**, with or without head+tail. Per protocol, neither 256-token variant was evaluated on test.

**Multi-task vs single-task (pre-registered test comparison):**

| Test metric (stress) | single-task | multi-task | Δ (MT − ST) [95 % CI] |
|---|---|---|---|
| PR-AUC | 0.875 | 0.852 | **−0.023 [−0.038, −0.008]** |
| macro-F1 | 0.783 | 0.778 | −0.005 [−0.021, +0.011] |
| F1 (stress class) | 0.798 | 0.796 | −0.003 [−0.017, +0.012] |

So the measured stress improvement comes from **separating the stress model** (PR-AUC 0.852 → 0.875), not from longer inputs. F1 at the default 0.5 threshold does not change, and single-task calibration is worse (ECE 0.062 vs 0.035), so the gain is in ranking quality, which matters most for screening thresholds. Mean false positives / false negatives on 715 test posts: multi-task 95.7 / 62.0, single-task 91.0 / 63.3.

## 4. Multi-task learning: no benefit anywhere; negative transfer on stress ranking

| Test task | single-task | multi-task | Δ (MT − ST) [95 % CI] |
|---|---|---|---|
| Sentiment macro-F1 | 0.606 | 0.603 | −0.002 [−0.014, +0.011] |
| — hi-Latn | 0.556 | 0.559 | +0.003 [−0.013, +0.020] |
| — en-Latn | 0.684 | 0.673 | −0.011 [−0.028, +0.009] |
| Emotion macro-F1 | 0.682 | 0.684 | +0.003 [−0.005, +0.010] |
| — en-Latn | 0.639 | 0.645 | +0.006 [−0.002, +0.012] |
| — hi-Deva | 0.770 | 0.768 | −0.002 [−0.012, +0.009] |
| Stress PR-AUC | 0.875 | 0.852 | **−0.023 [−0.038, −0.008]** |

Sharing one encoder across three unrelated datasets (different populations, platforms and annotators) gives no measurable gain on any task and hurts stress ranking. This is consistent with Pruksachatkun et al. (2020): transfer depends strongly on the task pairing. The practical benefit of the shared model is cost: one 238 M-parameter model instead of three, and ~127 vs ~150 CPU-minutes to train.

## 5. What explains the English–Hinglish gap? (partial answer)

* **Same platform, same annotation pipeline, different language.** Within the tweets-hindi test set, tweets that language-ID tags as English score far higher than those tagged Romanized Hindi: MuRIL 0.642 vs 0.534; HingRoBERTa 0.676 vs 0.596 (`reports/results/tweets_hindi_langid_stratified.json`). This points to language/script being associated with the gap beyond platform differences. Caveat: subsets are defined by a noisy classifier, and "English-looking" tweets may differ in other ways.
* **Pre-training data matters more than architecture.** A model pre-trained on Roman-script Hinglish closes ~27 % of the gap with no other change.
* **Not tested:** spelling normalisation, transliteration, error categories (§0). A labelled error taxonomy needs a human annotator; the sheet is ready (`scripts/export_annotation_sheet.py`).

## 6. Reliability: calibration and selective prediction

Temperature scaling (fit on validation) is applied to all runs; test ECE is in §1. Abstaining on the least-confident inputs, with the threshold chosen on **validation** for 80 % coverage and applied unchanged to test (`reports/results/selective_prediction.json`, `reports/figures/risk_coverage.png`):

| Model / task | Test macro-F1, all inputs | Test coverage | Test macro-F1, answered inputs |
|---|---|---|---|
| Single-task MuRIL, stress | 0.783 | 0.83 | 0.828 |
| Single-task MuRIL, sentiment | 0.606 | 0.77 | 0.649 |
| HingRoBERTa, sentiment | 0.647 | 0.74 | 0.702 |

Low confidence means *the model is uncertain about the text*. It says nothing about whether a person is or is not experiencing a mental-health problem.

## 7. Explanation faithfulness (Integrated Gradients)

Deletion test on the explanations the app serves (original multi-task MuRIL checkpoint; IG, 16 steps, zero baseline; delete k = ⌈20 %⌉ of content tokens; validation examples; `reports/results/faithfulness.json`). Drop in the predicted class's probability:

| Group | n | top-k deleted | random-k | bottom-k | top − random [95 % CI] | examples with top > random |
|---|---|---|---|---|---|---|
| English stress | 60 | 0.360 | 0.056 | −0.062 | **+0.305 [+0.222, +0.386]** | 92 % |
| English sentiment | 60 | 0.251 | 0.097 | −0.000 | **+0.155 [+0.090, +0.224]** | 68 % |
| Hinglish sentiment | 59 | 0.151 | 0.078 | 0.045 | +0.074 [+0.023, +0.125] | 58 % |

The IG highlights are **faithful for English stress**, moderately faithful for English sentiment, and **only weakly faithful for Hinglish**: for 42 % of Hinglish examples, deleting the "most important" tokens hurts no more than deleting random ones. The app's explanation caveat now states this. These are deletion-based comprehensiveness scores on 60 examples per group; sufficiency was not measured.

## 8. Limitations

* 3 seeds, not 5; the bootstrap covers test-sample noise only, and seed-level s.d. is reported separately.
* CPU fp32, 3-epoch budget; best epoch was often the last (under-training cannot be ruled out). The LR grid was not run.
* Stress = crowd-annotated *stress* in English Reddit posts; not clinical; not validated in any other language.
* The Hinglish data is small, partly noisy (English tweets labelled `hindi`, a single-annotator YouTube set, a suspicious validation split).
* The contamination check is exact-match only, on a re-uploaded corpus.
* Recorded `train_minutes` for resumed runs counts only the final process after a restart.
