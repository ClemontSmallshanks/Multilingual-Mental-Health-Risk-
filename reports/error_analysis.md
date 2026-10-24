# Error analysis

*Date: 2026-10-24. Numbers from `reports/results/error_breakdown.json` (seed-averaged probabilities on test, single-task runs), `tweets_hindi_langid_stratified.json`, `tweets_hindi_validation_anomaly.json`. Raw texts are never written to the repository; examples are referenced by anonymised `text_id`.*

## 1. Where sentiment errors are (test, seed-averaged predictions)

| | MuRIL error rate | HingRoBERTa error rate |
|---|---|---|
| English `en-Latn` (n = 870) | 0.309 | 0.282 |
| Hinglish `hi-Latn` (n = 1,343) | 0.436 | 0.370 |
| short texts, < 8 words (n = 483) | 0.433 | 0.371 |
| medium, 8–39 words (n = 1,707) | 0.372 | 0.323 |
| long, ≥ 40 words (n = 23) | 0.435 | 0.478 (n too small to interpret) |

Short texts are harder for both models; the long bucket has only 23 examples.

## 2. What kind of Hinglish errors HingRoBERTa removes

Seed-averaged confusion on `hi-Latn` test (rows = gold, columns = predicted):

| gold \ pred | MuRIL neg | MuRIL neu | MuRIL pos | | HingRoBERTa neg | HingRoBERTa neu | HingRoBERTa pos |
|---|---|---|---|---|---|---|---|
| negative (418) | 281 | 88 | 49 | | 278 | 86 | 54 |
| neutral (408) | 127 | 195 | 86 | | 80 | 241 | 87 |
| positive (517) | 131 | 104 | 282 | | 97 | 93 | 327 |

MuRIL **over-predicts negative** on Hinglish: 131 positive and 127 neutral comments become negative. HingRoBERTa reduces these to 97 and 80; negative recall is unchanged. The gain is fewer false "negative" readings of non-negative Hinglish text, not better detection of negative text. *Hypothesis (not tested):* MuRIL's sub-word segmentation of Romanized Hindi words, which it mostly saw in Devanagari or transliterated form, loses polarity cues. The manual error taxonomy below is the way to test this.

## 3. Language versus platform

Within one dataset (tweets-hindi test: same platform and annotation pipeline), split by our language-ID:

| subset | n | MuRIL macro-F1 | HingRoBERTa macro-F1 |
|---|---|---|---|
| detected Romanized Hindi | 642 | 0.534 ± 0.008 | 0.596 ± 0.044 |
| detected English | 223 | 0.642 ± 0.010 | 0.676 ± 0.006 |

A ~0.1 macro-F1 gap remains *inside* one source, so the English–Hinglish gap is not only a platform or annotation artefact. Caveat: the subsets come from a classifier (≈92 % recall on Hinglish YouTube, ≈99 % on English text; `reports/metrics/langid.json`).

## 4. Data problems found

1. **tweets-hindi validation split ≈ chance for every model** (0.35 vs ≈0.56–0.62 on its test split); cause unverified (label misalignment vs a hard split). It must not drive model selection until inspected.
2. **The "hindi" tweet config is Romanized and ~26 % English** by language-ID (found in Stage 1).
3. BRIGHTER English has no `disgust` labels; dev/test duplicates (handled; see `DATASET.md`).

## 5. Stress errors (English only)

Mean over 3 seeds, 715 test posts: single-task 91.0 FP / 63.3 FN, multi-task 95.7 FP / 62.0 FN. 97 % of stress test posts are ≥ 40 words, so short-text behaviour of the stress head is untested. Error rate by length: long 0.211, medium (n = 22) 0.136 (single-task).

## 6. Not done: manual error taxonomy (needs a human)

Automatic heuristics are not reported as error categories. Tooling for a manual taxonomy is ready:

```bash
python scripts/export_annotation_sheet.py --experiment C_ST_sentiment_hingroberta --n 200
# -> data/annotation/hinglish_val_errors_C_ST_sentiment_hingroberta.csv  (git-ignored: contains raw text)
```

Schema (mark 1 for every category that applies): `spelling_variant`, `code_switch`, `transliteration_ambiguity`, `negation_intensifier`, `slang`, `sarcasm_figurative`, `too_short_or_context`, `likely_label_error`, `english_only_text`, `other`, plus `notes`. Annotate twice, a week apart, to report intra-annotator Cohen's κ. Commit only `text_id` + category columns, never the text. The sheet samples **validation** errors so the test set stays untouched.
