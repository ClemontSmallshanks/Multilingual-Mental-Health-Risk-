"""Export a LOCAL (git-ignored) sheet of Hinglish VALIDATION errors for manual error-category annotation.

Stage 4 of reports/experiment_plan.md. The sheet contains raw text (needed to annotate) and is written to
data/annotation/ (git-ignored; tweet/YouTube text must not be redistributed). Only the filled-in category columns,
keyed by text_id, should be committed (see reports/error_analysis.md for the schema).

  python scripts/export_annotation_sheet.py --experiment C_ST_sentiment_hingroberta --n 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, PROCESSED_DIR, SENTIMENT_LABELS  # noqa: E402
from src.experiments.runner import runs_dir  # noqa: E402

CATEGORIES = ["spelling_variant", "code_switch", "transliteration_ambiguity", "negation_intensifier", "slang",
              "sarcasm_figurative", "too_short_or_context", "likely_label_error", "english_only_text", "other"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="C_ST_sentiment_hingroberta")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    S = pd.read_csv(PROCESSED_DIR / "sentiment.csv", keep_default_na=False).set_index("text_id")
    frames = [pd.read_csv(f) for f in sorted((runs_dir() / a.experiment).glob("seed_*/predictions_val_sentiment.csv"))]
    if not frames:
        sys.exit("no validation predictions found")
    P = np.mean([f[["p_negative", "p_neutral", "p_positive"]].to_numpy() for f in frames], axis=0)  # seed-averaged
    d = frames[0][["text_id", "source", "language", "y"]].copy()
    d["pred"] = P.argmax(1)
    err = d[(d.language == "hi-Latn") & (d.pred != d.y)]
    err = err.sample(n=min(a.n, len(err)), random_state=a.seed)
    sheet = pd.DataFrame({"text_id": err.text_id, "source": err.source, "text": S.loc[err.text_id, "text"].values,
                          "gold": [SENTIMENT_LABELS[i] for i in err.y], "pred": [SENTIMENT_LABELS[i] for i in err.pred]})
    for c in CATEGORIES:
        sheet[c] = ""
    sheet["notes"] = ""
    out = DATA_DIR / "annotation"
    out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out / f"hinglish_val_errors_{a.experiment}.csv", index=False)
    print(f"wrote {len(sheet)} rows to {out} (git-ignored). Mark each category column with 1 where it applies.")
