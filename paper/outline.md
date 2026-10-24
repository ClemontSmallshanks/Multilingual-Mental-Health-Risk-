# Paper draft outline

**Working title:** *Closing the Romanized-Hindi Gap: Pre-training Data, Not Multi-Task Sharing, Drives Multilingual Sentiment and Stress-Language Classification*

Status legend: ✅ measured (number from `runs/`) · ⏳ running/pending · ❌ not run (reason given). No anticipated numbers appear here.

## 1. Introduction
* Motivation: multilingual screening classifiers for affect and stress-related language are increasingly built on one shared multilingual encoder; Romanized/code-mixed Hindi ("Hinglish") is common online but under-resourced.
* Research question: what explains the English–Hinglish gap, and which interventions reliably close it?
* Contributions (only those supported by evidence):
  1. ✅ A reproducible, multi-seed, pre-registered re-evaluation of MuRIL on three public datasets (Dreaddit, UMSAB tweets + Hinglish YouTube, BRIGHTER), with frozen splits and a test-set discipline.
  2. ✅ Hinglish-specific pre-training (HingRoBERTa) improves Hinglish sentiment macro-F1 by +0.056 [+0.032, +0.079] on test with no English loss, closing 27 % of the gap; contamination check included.
  3. ✅ A controlled single- vs multi-task ablation: no task benefits from sharing; stress PR-AUC suffers negative transfer (−0.023 [−0.038, −0.008]).
  4. ✅ Negative result: head+tail truncation for long Reddit posts does not help.
  5. ✅ Integrated-Gradients explanations are much less faithful for Hinglish than for English.
  6. ✅ Data findings: a near-chance validation split in a widely used Hindi tweet benchmark config; Romanized and ~26 % English content in the "hindi" config.
* Ethics framing: stress *language* screening, not diagnosis (Chancellor & De Choudhury, 2020; Benton et al., 2017; Harrigian et al., 2020).

## 2. Related work
Multilingual / Indic encoders (Conneau et al., 2020; Khanuja et al., 2021); code-mixed NLP (Khanuja et al., 2020; Patwa et al., 2020; Nayak & Joshi, 2022); transliteration (Roark et al., 2020; Madhani et al., 2022); multi-task transfer (Liu et al., 2019; Pruksachatkun et al., 2020; Yu et al., 2020); fine-tuning variance (Dodge et al., 2020; Mosbach et al., 2021); calibration and selective prediction (Guo et al., 2017; Desai & Durrett, 2020; Geifman & El-Yaniv, 2017; Hendrycks & Gimpel, 2017); explanation faithfulness (Sundararajan et al., 2017; DeYoung et al., 2020; Jacovi & Goldberg, 2020); mental-health NLP validity (Harrigian et al., 2020). Full verified list: `reports/experiment_plan.md`.

## 3. Data
Dreaddit (English stress), UMSAB English + "hindi" tweets, Hinglish YouTube comments, BRIGHTER (English, Hindi). Frozen split hashes (`splits/`), dedup/near-dup removal, group splits. Population differences between tasks; label definitions; `DATASET.md`.

## 4. Methods
* Encoders: MuRIL-base (multi-task: shared encoder + shared layer + 3 heads; single-task: same, one head), HingRoBERTa (single-task sentiment).
* Training: AdamW, encoder LR 3e-5, head LR 5e-4, 6 % warmup, 3 epochs, early stopping on validation macro-F1 (patience 2), batch 16, fp32 CPU; seeds 13/21/42.
* Calibration: temperature scaling on validation; emotion thresholds tuned on validation.
* Statistics: paired bootstrap over test examples on seed-averaged metrics (1,000 resamples); seed s.d. reported separately (Berg-Kirkpatrick et al., 2012; Dror et al., 2018).
* Pre-registration: which runs may touch test (`experiment_plan.md` §7); HingRoBERTa tested once after validation selection.

## 5. Results
* Table 1 ✅ headline test results (`reports/results_summary.md` §1).
* Table 2 ✅ HingRoBERTa vs MuRIL by language and source (§2).
* Table 3 ✅ multi- vs single-task ablation (§4).
* Table 4 ✅ truncation (validation): head+tail-256 and head-256 both no better than 128 (§3).
* Figure 1 ✅ risk–coverage curves (`reports/figures/risk_coverage.png`).
* Table 5 ✅ language-ID-stratified tweets-hindi analysis (`error_analysis.md` §3).
* Table 6 ✅ IG faithfulness: faithful for English stress (+0.305 [+0.222, +0.386]), weak for Hinglish sentiment (+0.074 [+0.023, +0.125], 58 % of examples) (§7).

## 6. Analysis
✅ Confusion shift on Hinglish (MuRIL over-predicts negative; HingRoBERTa fixes non-negative→negative errors). ✅ Within-source language gap. ❌ Manual error taxonomy (needs a human annotator; tooling ready). ❌ Normalisation / transliteration (not implemented).

## 6b. Comparison with published results and conclusion
✅ Written up in [`reports/conclusion.md`](../reports/conclusion.md): stress on par with Dreaddit BERT-base (F1 0.798 vs 0.8065, same test set); Hinglish pre-training gain replicates Nayak & Joshi (2022) (+5.6 vs their +4.86); UMSAB Hindi tweets above published XLM-Twitter (0.616 vs 56.39, different training mix); emotion below SemEval-2025 RoBERTa baselines.

## 7. Limitations
3 seeds; CPU 3-epoch budget, no LR search; English-only stress; small/noisy Hinglish data and suspicious validation split; exact-match contamination check on a re-uploaded corpus; single annotator data; stress ≠ clinical status.

## 8. Ethics statement
Screening/research only; no diagnosis; no raw text released; privacy-preserving logging; crisis-resource policy (no phone numbers, directories only).

## Appendix
Reproducibility (`reports/reproducibility.md`), configs (`configs/cpu/`), per-run artefacts (`runs/`), contamination check (`reports/results/hingcorpus_overlap.json`).
