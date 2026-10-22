"""Freeze (or --check) split membership: splits/<task>.csv holds text_id, split, sha1(text); never the text."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR, SPLITS_DIR  # noqa: E402
from src.data.splits import freeze, verify  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify processed data against the frozen split (exit 1 on mismatch)")
    a = ap.parse_args()
    if a.check:
        for t in ("sentiment", "emotion", "risk"):
            verify(t, pd.read_csv(PROCESSED_DIR / f"{t}.csv", keep_default_na=False), SPLITS_DIR)
            print(f"{t}: matches frozen split")
    else:
        if (SPLITS_DIR / "MANIFEST.json").exists():
            sys.exit("splits/ already frozen. Refusing to overwrite; delete it deliberately if you really mean to.")
        print(json.dumps(freeze(PROCESSED_DIR, SPLITS_DIR), indent=2))
