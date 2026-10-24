# Conclusion: what this project improved, and how it compares with published research

*Date: 2026-10-24. Our numbers: `reports/results_summary.md` (test sets, mean over 3 seeds, paired-bootstrap 95 % CIs). Published numbers were read from the papers' own result tables (table numbers given) and are **context, not head-to-head comparisons**, unless the test set and protocol are the same (stated per row).*

## 1. What we improved (measured, test set, pre-registered)

| # | Improvement | Before → after | Evidence | Research basis |
|---|---|---|---|---|
| 1 | **Hinglish sentiment**: replace MuRIL with a Hinglish-pre-trained encoder (HingRoBERTa) | `hi-Latn` macro-F1 **0.556 → 0.611** (+0.056 [+0.032, +0.079]); all sentiment 0.606 → 0.647; English unchanged | gain on *both* Hinglish sources (tweets +0.054, YouTube +0.071); English–Hinglish gap −27 %; 0 test texts found in HingRoBERTa's 52.9 M-line pre-training corpus | Nayak & Joshi (2022), Khanuja et al. (2020) |
| 2 | **Stress detection**: train the stress classifier on its own instead of sharing a multi-task model | PR-AUC **0.852 → 0.875** (+0.023 [+0.008, +0.038]); ROC-AUC 0.854 → 0.867 | F1 unchanged (0.796 → 0.798); calibration worse (ECE 0.035 → 0.062) | Pruksachatkun et al. (2020); Liu et al. (2019) |
| 3 | **Reliability**: abstain on low-confidence inputs (threshold chosen on validation) | stress macro-F1 0.783 → **0.828** on the 83 % of inputs answered; HingRoBERTa sentiment 0.647 → 0.702 at 74 % | risk–coverage curves in `reports/figures/risk_coverage.png` | Geifman & El-Yaniv (2017); Hendrycks & Gimpel (2017) |
| 4 | **Research integrity / engineering** | single seed, unpinned, unverified → 3 seeds, frozen split hashes, pinned packages, resumable runs, pre-registered test use; original run reproduced *exactly* | `reports/reproducibility.md` | Dodge et al. (2020); Dror et al. (2018); Berg-Kirkpatrick et al. (2012) |

**What did not improve (reported, not hidden):**
* Longer stress inputs: head+tail truncation to 256 tokens (Sun et al., 2019) was **worse** on validation (macro-F1 −0.020 [−0.037, −0.003]); plain 256-token input made no difference.
* Multi-task sharing gave **no gain on any task** and hurt stress ranking.
* Emotion was not improved and stays **below** published baselines (see §2).
* The served explanations (Integrated Gradients) are faithful for English stress but only **weakly faithful for Hinglish**.

## 2. Comparison with published results

| Task / dataset | Published result (source) | Ours (test, 3-seed mean) | Comparable? | Reading |
|---|---|---|---|---|
| **Stress, Dreaddit** (Turcan & McKeown, 2019, Table 6) | BERT-base F1 **0.8065**; best LogReg F1 0.7980 | single-task MuRIL F1 **0.798**; multi-task 0.796 | **Largely yes**: same official 715-post test set; our training set is smaller (15 % held out for validation) | **On par** with the original paper's results, not above them |
| **Hinglish sentiment, GLUECoS** (Nayak & Joshi, 2022, Roman-script table) | HingRoBERTa F1 66.43 vs XLM-R 61.57 (**+4.86**) | HingRoBERTa vs MuRIL on our Hinglish data: **+5.6** macro-F1 | No: different dataset and baseline encoder | **Independent replication of the effect**: Hinglish pre-training helps by a similar margin on new data |
| **Hindi (Romanized) tweets, UMSAB** (Barbieri et al., 2022, Table 7, one multilingual model) | XLM-R 53.39 ± 2.00, XLM-Twitter **56.39 ± 1.60** | MuRIL 0.561, **HingRoBERTa 0.616** (tweets-hindi test) | Partly: same test file (866 of 870 tweets after dedup), but we train on 2 languages + YouTube Hinglish, with different hyper-parameters | HingRoBERTa is **≈5 points above** the published XLM-Twitter result; treat as indicative only |
| **English tweets, UMSAB** (same table) | XLM-R 68.52 ± 1.42, XLM-Twitter 70.63 ± 1.04 | MuRIL 0.684, HingRoBERTa 0.704 | Partly (same test file, different training mix) | **On par** |
| **Emotion, BRIGHTER English** (SemEval-2025 Task 11, Table 2; BRIGHTER Table 5) | RoBERTa baseline 0.708; top system 0.823; XLM-R 67.30 | 0.639 (single-task), 0.645 (multi-task) | Partly: we deduplicated the test file (2,759 vs 2,767), and English has no `disgust` labels | **Below** published baselines |
| **Emotion, BRIGHTER Hindi** (same sources) | RoBERTa baseline 0.855; top system 0.926; RemBERT 85.51 | 0.770 / 0.768 | Partly (same 1,010 test texts after dedup) | **Below** published baselines |

**Honest summary:** the stress classifier matches the original Dreaddit paper. The Hinglish sentiment model shows a real, statistically supported improvement that replicates the HingBERT paper's finding on independent data, and it appears to exceed the published UMSAB Hindi result under a different training setup. Emotion is the weakest component and the clearest target for future work: it was trained for only 3 epochs on CPU, with less training data because validation was carved from train.

## 3. Agreement and disagreement with the literature

* **Pre-training data matters for code-mixed text** (Nayak & Joshi, 2022; Khanuja et al., 2020): **confirmed**. The gain concentrates on non-negative Hinglish texts that MuRIL misread as negative (`reports/error_analysis.md` §2).
* **Multi-task transfer depends on the task pairing** (Pruksachatkun et al., 2020): **consistent**. Three tasks from unrelated datasets showed no positive transfer; stress showed negative transfer. MT-DNN-style gains (Liu et al., 2019) did not appear in this setting.
* **Head+tail truncation helps long documents** (Sun et al., 2019): **not reproduced for Reddit posts**. Their gains were on long reviews and news at a 510-token budget; for ~100-token Reddit posts, the extra context did not help.
* **Seed variance is large** (Dodge et al., 2020; Mosbach et al., 2021): **confirmed**. Seed s.d. reaches 0.029 (stress PR-AUC, multi-task) and 0.044 (HingRoBERTa on a Hinglish subset), the same size as several single-seed "improvements".
* **Temperature scaling improves calibration** (Guo et al., 2017; Desai & Durrett, 2020): applied throughout; the better-ranking single-task stress model is *less* calibrated, so ranking and calibration should be reported separately.
* **Explanations must be tested, not assumed** (Jacovi & Goldberg, 2020; DeYoung et al., 2020): the deletion test shows faithfulness is **language-dependent**: strong for English stress, weak for Hinglish. This is a new, small-scale observation worth following up.
* **Mental-health models generalise poorly across datasets** (Harrigian et al., 2020): consistent with our decision not to claim any stress capability outside English Dreaddit.

## 4. The research paper

A paper draft is in **[`paper/outline.md`](../paper/outline.md)**. Working title: *"Closing the Romanized-Hindi Gap: Pre-training Data, Not Multi-Task Sharing, Drives Multilingual Sentiment and Stress-Language Classification"*. Its main contribution is the controlled finding that Hinglish-specific pre-training, not shared multi-task training, closes a significant part of the English–Hinglish gap, together with the multi-task ablation, the truncation negative result, language-dependent explanation faithfulness, and two data-quality findings in a public benchmark. All 35 references were verified against arXiv / ACL Anthology / Crossref (`reports/experiment_plan.md`, References).

## 5. Limitations and next steps

Limitations: 3 seeds on CPU with a 3-epoch budget and no learning-rate search; stress validated in English only and not a clinical label; small, partly noisy Hinglish data with a suspicious validation split; exact-match contamination check only.

Next steps, in order of expected value:
1. Train emotion longer, and consider RemBERT, which BRIGHTER reports as strongest.
2. Run the learning-rate grid and 5 seeds on a GPU.
3. Inspect the tweets-hindi validation labels by hand.
4. Complete the manual Hinglish error taxonomy (sheet ready).
5. Test transliteration (Madhani et al., 2022).
6. Serve HingRoBERTa for Hinglish sentiment in the app.

## References cited here

Barbieri, Espinosa Anke & Camacho-Collados (2022) LREC · Berg-Kirkpatrick, Burkett & Klein (2012) EMNLP-CoNLL · Desai & Durrett (2020) EMNLP · DeYoung et al. (2020) ACL · Dodge et al. (2020) arXiv:2002.06305 · Dror et al. (2018) ACL · Geifman & El-Yaniv (2017) arXiv:1705.08500 · Guo et al. (2017) ICML · Harrigian, Aguirre & Dredze (2020) Findings of EMNLP · Hendrycks & Gimpel (2017) ICLR · Jacovi & Goldberg (2020) ACL · Khanuja et al. (2020) GLUECoS, ACL · Liu et al. (2019) MT-DNN, ACL · Madhani et al. (2022) Aksharantar · Mosbach et al. (2021) ICLR · Muhammad et al. (2025) BRIGHTER, ACL · Muhammad et al. (2025) SemEval-2025 Task 11 · Nayak & Joshi (2022) arXiv:2204.08398 · Pruksachatkun et al. (2020) ACL · Sun et al. (2019) arXiv:1905.05583 · Turcan & McKeown (2019) LOUHI. Full entries with identifiers: `reports/experiment_plan.md`.
