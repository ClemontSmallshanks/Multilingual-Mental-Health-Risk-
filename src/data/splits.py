"""Frozen split membership.

`splits/<task>.csv` records, for every processed example, its text_id, split and a SHA-1 of the (cleaned) text, but
never the text itself. `splits/MANIFEST.json` stores a SHA-256 per file. Loading data verifies the processed tables
against these files, so a changed download, preprocessing step or library version cannot silently move examples
between train/val/test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class SplitMismatchError(RuntimeError):
    pass


def _text_hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"text_id": df.text_id, "split": df.split, "text_sha1": df.text.map(_text_hash)})


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(processed_dir: Path, splits_dir: Path, tasks=("sentiment", "emotion", "risk")) -> dict:
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for t in tasks:
        df = pd.read_csv(processed_dir / f"{t}.csv", keep_default_na=False)
        out = splits_dir / f"{t}.csv"
        split_table(df).to_csv(out, index=False)
        manifest[t] = {"sha256": _file_sha256(out), "n": int(len(df)), "by_split": df.split.value_counts().to_dict()}
    (splits_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify(task: str, df: pd.DataFrame, splits_dir: Path) -> bool:
    """Raise SplitMismatchError if `df` (a full processed table) differs from the frozen split. Returns False if no
    frozen split exists (nothing to check)."""
    f = splits_dir / f"{task}.csv"
    if not f.exists():
        return False
    man = json.loads((splits_dir / "MANIFEST.json").read_text())
    if _file_sha256(f) != man[task]["sha256"]:
        raise SplitMismatchError(f"{f} does not match MANIFEST.json (file edited?)")
    frozen = pd.read_csv(f, keep_default_na=False)
    cur = split_table(df).reset_index(drop=True)
    if len(cur) != len(frozen) or not cur.equals(frozen):
        diff = (cur.merge(frozen, on="text_id", how="outer", suffixes=("", "_frozen"), indicator=True)
                   .query("_merge != 'both' or split != split_frozen or text_sha1 != text_sha1_frozen"))
        raise SplitMismatchError(
            f"processed {task} data differs from the frozen split in {len(diff)} rows "
            f"(e.g. {diff.text_id.head(3).tolist()}). Re-run scripts/prepare_data.py with the documented versions, "
            f"or investigate before training; never re-freeze to make this error go away.")
    return True
