# Experiment plan: literature-backed upgrade

*Status (updated 2026-10-22): Stage 2 framework implemented (`configs/`, `scripts/run_experiment.py`, `scripts/aggregate_runs.py`, `notebooks/colab_experiments.ipynb`; see `reproducibility.md`). **No experiment below has been run yet.** No numbers in this file are results. Baseline numbers quoted are the existing single-seed MuRIL results from `reports/metrics/` (seed 13), verified in `research_audit.md`.*

## 0. Research question

> What explains the performance gap between English and Romanized Hindi/Hinglish text classification, and which interventions reliably improve performance without compromising evaluation integrity?

The two **major improvements** targeted, each tied to published evidence and a falsifiable success criterion:

| | Target | Current (1 seed) | Literature basis | Success criterion (pre-registered) |
|---|---|---|---|---|
| **M1** | English stress detection (Dreaddit) | F1 0.807 · PR-AUC 0.830 · ROC-AUC 0.848 | Truncation strategies [Sun 2019]; fine-tuning stability / LR / longer training [Mosbach 2021; Dodge 2020] | Mean over 5 seeds improves **PR-AUC and F1** over the 5-seed baseline, with a paired bootstrap 95 % CI on the difference excluding 0 |
| **M2** | Hinglish (hi-Latn) sentiment | macro-F1 0.557 vs English 0.696 | Hinglish-specific pre-training [Nayak & Joshi 2022]; code-switching benchmarks [Khanuja 2020 (GLUECoS); Patwa 2020]; transliteration [Madhani 2022; Roark 2020] | Mean over 5 seeds improves **hi-Latn macro-F1** over 5-seed MuRIL, CI excluding 0, **without** lowering English macro-F1 by more than its seed s.d. |

A negative result (no significant gain) is an acceptable, reportable outcome. It will not be hidden.

---

## 1. Evaluation protocol (applies to every experiment)

* **Splits frozen.** The existing `data/processed` split membership is exported to `splits/*.csv` (text_id, split; no text) with a SHA-256. Every run asserts it. Split membership will never be changed to improve results.
* **Test set discipline.** All tuning and selection on `val`. The test set is evaluated **once per pre-registered final configuration** (listed in §7). Exploratory runs log validation metrics only.
* **Seeds.** Exploration: 1–2 seeds. Final comparisons: 5 seeds (13, 21, 42, 55, 77). Report mean ± s.d. and per-seed values. Motivation: seed variance in BERT fine-tuning is comparable to many claimed improvements [Dodge 2020; Mosbach 2021].
* **Statistics.** Paired bootstrap over test examples (1,000 resamples; same resamples for both systems) for the difference in each metric [Berg-Kirkpatrick 2012; Dror 2018]. Also report seed-level comparison (mean diff ± s.d.). Report all comparisons run, not just significant ones.
* **Metrics per task:**
  * Sentiment: macro-F1 (headline), per-class P/R/F1, confusion matrix, **reported separately for en-Latn, tweets-hindi, youtube-hinglish**; pooled only as secondary.
  * Emotion: macro-F1 over annotated labels (masked), per-label P/R/F1, per language; threshold per label tuned on val. The "error rate" metric is renamed `exact_set_mismatch_rate` and per-label error rate is added (fixes audit A5).
  * Stress: F1 (positive = stress), precision, recall, PR-AUC, ROC-AUC, FP/FN counts, ECE (15 equal-width bins), Brier score.
* **Run metadata.** Every run saves the resolved config, seed, model id + revision, package versions, device, train/inference time, peak memory, val/test predictions with probabilities.

---

## 2. Stage 2: reproducibility and multi-seed baseline (prerequisite)

| ID | Experiment | Cost (CPU, 4 cores) |
|---|---|---|
| R0 | ✅ **Done:** pinned requirements, `configs/`, per-run directories, frozen split + automatic check, deterministic flag, emotion error-rate split into per-label vs exact-set | — |
| R1 | TF-IDF baseline × 5 seeds (seed only affects nothing deterministic, so 1 run + note) | minutes |
| R2 | MuRIL multi-task, current config, **× 5 seeds** | ~10 h CPU · ~1 h GPU (est.) |

R2 becomes **the** baseline for every later comparison. The existing single-seed numbers are kept for reference but are not the comparison point.

---

## 3. Stage 3: Major improvement 1, stress detection (M1)

**Hypotheses** (each can fail):

* **H1a (truncation).** Truncating 23.6 % of Dreaddit inputs at 128 tokens loses signal. Using `max_len=256`, or head+tail truncation, improves stress F1/PR-AUC. *Basis:* Sun et al. (2019, §5.3.1, Table 2): for texts longer than BERT's limit, keeping the first 128 + last 382 tokens ("head+tail") beat head-only and tail-only on IMDb and Sogou. Their texts were long reviews/news at a 510-token budget; whether this transfers to short Reddit posts at 256 tokens is exactly what E1.2 tests.
* **H1b (optimisation).** The current schedule (3 epochs, head LR 16.7× encoder LR) is not optimal. Rising validation loss at epoch 3 with rising F1 (audit A2) suggests over-confidence, not convergence. *Basis:* Mosbach et al. (2021, §6) trace fine-tuning instability on small datasets to optimisation difficulties (vanishing gradients) and recommend "small learning rates with bias correction" and to "increase the number of iterations considerably"; their baseline is LR 2e-5, 10 % warmup, 20 epochs. (PyTorch `AdamW`, used here, already applies bias correction, so the untested factor is the training budget.) Dodge et al. (2020) show that the random seed alone (weight initialisation and data order) causes substantial variance, and that early stopping matters on small datasets.

| ID | Variant (only one factor changes vs R2) | Selection |
|---|---|---|
| E1.1 | `max_len_risk=256` (head truncation) | val |
| E1.2 | head+tail truncation (first 64 + last 190 tokens of a 256 budget, i.e. Sun et al.'s ~25 % head ratio) | val |
| E1.3 | Grid: encoder LR ∈ {1e-5, 2e-5, 3e-5} × head LR ∈ {5e-5, 1e-4, 2e-4} (9 cells), 2 seeds, up to 10 epochs (Mosbach-style longer budget), early stop on val macro-F1 with patience 3 **and** report val loss | val |
| E1.4 | Best of E1.1–E1.3 combined, × 5 seeds → test once | **final** |

*Cost:* E1.1–E1.2 ~4 h CPU (1 seed each); E1.3 is 18 runs. On CPU that is impractical, so it should be **run on a GPU, or reduced to 3 cells chosen a priori** (current, {2e-5, 1e-4}, {1e-5, 5e-5}).

---

## 4. Stage 3: Major improvement 2, the Hinglish gap (M2)

### 4.1 First, decompose the gap (no training needed)

The existing gap (0.696 vs 0.557) confounds language with platform, annotator and contamination. Before any intervention:

1. **De-contaminate.** Run the language-ID classifier on `tweets-hindi` and report metrics on (a) all, (b) only rows predicted `hi-Latn`. Measure how much English tweets shift the result. Also hand-verify a random sample of 100 val rows to estimate classifier precision. Heuristic labels are **not** treated as ground truth.
2. **Stratify by source.** Report tweets-english vs tweets-hindi (same platform and annotation pipeline, different language) separately from youtube-hinglish (different platform). The **tweets-english vs tweets-hindi** contrast is the cleanest available estimate of a language effect.
3. **Label-noise check.** Cleanlab-style confident-learning is *not* added. Instead, 100 val disagreements are hand-labelled by the author with a written guideline to estimate annotation noise in each source.

### 4.2 Interventions (each vs the 5-seed MuRIL baseline, same splits)

| ID | Intervention | Literature basis | Notes / risks |
|---|---|---|---|
| E2.1 | **Hinglish-specific encoder**: `l3cube-pune/hing-roberta` (CC-BY-4.0) single-task sentiment | Nayak & Joshi (2022) pre-trained HingBERT/HingMBERT/HingRoBERTa on L3Cube-HingCorpus (52.93 M Roman-script Hindi-English sentences scraped from Twitter) and evaluated them on GLUECoS tasks including code-mixed sentiment | Different tokenizer; same splits/labels. The pre-training corpus is Twitter-derived, so overlap with our `tweets-hindi` test texts is possible. Masked-LM pre-training on test text is a contamination risk, so the overlap check is mandatory |
| E2.2 | **XLM-R base** (MIT) single-task | Conneau et al. (2020); used as code-switching baseline in GLUECoS (Khanuja et al., 2020) | Controls for "MuRIL vs any multilingual encoder" |
| E2.3 *(not yet implemented)* | **Transliteration to Devanagari** of hi-Latn input (IndicXlit / Aksharantar models, `ai4bharat-transliteration`), then MuRIL | Madhani et al. (2022) (Aksharantar); Roark et al. (2020) (Dakshina) document romanisation variability; MuRIL was pre-trained with transliterated data (Khanuja et al., 2021) | English words in code-mixed text may be wrongly transliterated. Variants: (a) all tokens, (b) only tokens the language-ID marks Hindi. Original text kept; transformation logged. Choice made on **val** only |
| E2.4 *(not yet implemented)* | **Conservative spelling normalisation** (lower-case, character-repeat squashing "sooo"→"soo", fixed mapping list built from **train/val only**) | Motivated by romanisation variability [Roark 2020]; a deliberately weak baseline for E2.3 | Mapping list frozen before test evaluation |

*Selection rule:* the single best variant on val goes to a 5-seed test evaluation. The others are reported with val numbers only, to avoid test reuse.

### 4.3 Error taxonomy (Stage 4)

Hand-annotate 200 **validation** errors (not test) with a written schema: spelling variant, code-switch point, transliteration ambiguity, negation/intensifier, slang, sarcasm/figurative, too short or context-dependent, likely label error. Two passes a week apart to estimate intra-annotator agreement (Cohen's κ). Categories are reported with counts and κ, not as automatic heuristics.

---

## 5. Stage 3: multi-task vs single-task ablation (P4)

| ID | Setup | Basis |
|---|---|---|
| E3.1 | MuRIL single-task × 3 tasks (same splits, same tuned LR/epochs, same seeds) vs R2 multi-task | MT-DNN (Liu et al., 2019) reports gains from multi-task fine-tuning on GLUE. Pruksachatkun et al. (2020) studied 110 intermediate→target task pairs and found the benefit depends strongly on the pairing, so transfer between our three unrelated datasets cannot be assumed to be positive |
| E3.2 *(only if E3.1 shows negative transfer)* | Gradient-conflict mitigation (PCGrad) or loss re-weighting | Yu et al. (2020) |

Report per-task Δ (single − multi) with paired CIs. Negative transfer will be reported as a finding.

---

## 6. Stage 5: reliability and explanations (P5)

| ID | Experiment | Basis |
|---|---|---|
| E4.1 | Calibration: temperature scaling fitted on a **separate calibration half of val** (fixes audit A7); reliability diagrams; ECE/Brier per task and per language | Guo et al. (2017); Desai & Durrett (2020) studied calibration of pre-trained transformers in- and out-of-domain |
| E4.2 | Selective prediction: risk–coverage curves using max-softmax confidence; abstention threshold chosen on val for a target coverage | Geifman & El-Yaniv (2017); Hendrycks & Gimpel (2017) max-softmax baseline |
| E4.3 | OOD probe: confidence on texts from a task's *other* datasets (e.g. sentiment head on Dreaddit), reported as a distribution-shift check, not accuracy | Hendrycks & Gimpel (2017); Harrigian et al. (2020) show mental-health models generalise poorly across datasets |
| E4.4 | IG faithfulness: delete top-k vs random-k vs bottom-k tokens and measure the drop in predicted probability (comprehensiveness) and the probability kept with only the top-k tokens (sufficiency), English vs Hinglish | Sundararajan et al. (2017); DeYoung et al. (2020) (ERASER metrics); Jacovi & Goldberg (2020) on faithfulness vs plausibility |

If IG fails the deletion test (top-k drop not greater than random-k), the UI label is changed from "model attribution" to "gradient heat-map (not validated as faithful)".

---

## 7. Pre-registered final test evaluations

Only these configurations touch the test set, each × 5 seeds:

1. R2: MuRIL multi-task, current config (baseline)
2. E1.4: best stress configuration (M1)
3. Best of E2.1–E2.4 on val (M2)
4. E3.1: single-task MuRIL (ablation)
5. TF-IDF baseline (deterministic, 1 run)

## 8. Compute budget

| Hardware | One MuRIL run (3 tasks, 3 epochs) | Feasible scope |
|---|---|---|
| Current CPU (4 cores) | ~2 h (measured) | R2 + E1.1/E1.2 + reduced E1.3 + E2.1 + E3.1 at 2–3 seeds ≈ 40–60 h. Tight. |
| Single consumer/cloud GPU (e.g. T4) | ~10–20 min **(estimate, not measured)** | Full plan with 5 seeds |

If no GPU is available, the cheaper valid alternative is to run **single-task** experiments only for the task under study (stress-only runs are ~25 % of a multi-task run), use 3 seeds, and say so explicitly.

## 9. Deliverables (to be produced as stages complete)

`reports/research_audit.md` (done) · `reports/experiment_plan.md` (this file) · `reports/results_summary.md` · `reports/error_analysis.md` · `reports/reproducibility.md` · `reports/results/experiments.csv` (one row per run) · `paper/outline.md` (methods, results placeholders marked **NOT YET RUN**, limitations, related work).

---

## References (all verified against arXiv / ACL Anthology / Crossref on 2026-10-22)

**Datasets**
* Turcan, E. & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. *LOUHI Workshop*. arXiv:1911.00133
* Barbieri, F., Espinosa Anke, L. & Camacho-Collados, J. (2022). XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond. *LREC 2022*. ACL Anthology 2022.lrec-1.27
* Muhammad, S. H. et al. (2025). BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages. *ACL 2025*. arXiv:2502.11926
* Muhammad, S. H. et al. (2025). SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. arXiv:2503.07269
* Patwa, P. et al. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. *SemEval-2020*. arXiv:2008.04277

**Models**
* Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730
* Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL 2020*. arXiv:1911.02116
* Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692
* Nayak, R. & Joshi, R. (2022). L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models. arXiv:2204.08398
* Loureiro, D. et al. (2022). TimeLMs: Diachronic Language Models from Twitter. *ACL 2022 Demo*. arXiv:2202.03829

**Code-mixing and transliteration**
* Khanuja, S. et al. (2020). GLUECoS: An Evaluation Benchmark for Code-Switched NLP. *ACL 2020*. arXiv:2004.12376
* Madhani, Y. et al. (2022). Aksharantar: Open Indic-language Transliteration Datasets and Models for the Next Billion Users. *EMNLP Findings*. arXiv:2205.03018
* Roark, B. et al. (2020). Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset. *LREC 2020*. arXiv:2007.01176

**Fine-tuning, multi-task learning**
* Sun, C. et al. (2019). How to Fine-Tune BERT for Text Classification? arXiv:1905.05583
* Mosbach, M. et al. (2021). On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines. *ICLR 2021*. arXiv:2006.04884
* Dodge, J. et al. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305
* Liu, X. et al. (2019). Multi-Task Deep Neural Networks for Natural Language Understanding (MT-DNN). *ACL 2019*. arXiv:1901.11504
* Pruksachatkun, Y. et al. (2020). Intermediate-Task Transfer Learning with Pretrained Models for Natural Language Understanding: When and Why Does It Work? *ACL 2020*. arXiv:2005.00628
* Yu, T. et al. (2020). Gradient Surgery for Multi-Task Learning (PCGrad). *NeurIPS 2020*. arXiv:2001.06782

**Calibration, uncertainty, OOD**
* Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv:1706.04599
* Desai, S. & Durrett, G. (2020). Calibration of Pre-trained Transformers. *EMNLP 2020*. arXiv:2003.07892
* Geifman, Y. & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. arXiv:1705.08500
* Hendrycks, D. & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*. arXiv:1610.02136

**Explanations**
* Sundararajan, M. et al. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). arXiv:1703.01365
* DeYoung, J. et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL 2020*. arXiv:1911.03429
* Jacovi, A. & Goldberg, Y. (2020). Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness? *ACL 2020*. arXiv:2004.03685

**Statistics and reporting**
* Berg-Kirkpatrick, T., Burkett, D. & Klein, D. (2012). An Empirical Investigation of Statistical Significance in NLP. *EMNLP-CoNLL 2012*. ACL Anthology D12-1091
* Dror, R. et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *ACL 2018*. ACL Anthology P18-1128
* Gebru, T. et al. (2018/2021). Datasheets for Datasets. *CACM*. arXiv:1803.09010
* Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT\* 2019*. arXiv:1810.03993

**Mental-health NLP: validity and ethics**
* Harrigian, K., Aguirre, C. & Dredze, M. (2020). Do Models of Mental Health Based on Social Media Data Generalize? *Findings of EMNLP 2020*. ACL Anthology 2020.findings-emnlp.337
* Chancellor, S. & De Choudhury, M. (2020). Methods in Predictive Techniques for Mental Health Status on Social Media: A Critical Review. *npj Digital Medicine* 3, 24.
* Benton, A., Coppersmith, G. & Dredze, M. (2017). Ethical Research Protocols for Social Media Health Research. *First ACL Workshop on Ethics in NLP*. ACL Anthology W17-1612

*Note on how claims are attributed:* Sun et al. (2019) and Mosbach et al. (2021) claims were checked against the full text (sections cited above). All other one-line descriptions are summaries of the abstracts only. Before quoting any specific number from a paper in the write-up, read the full paper and cite the section/table.
