"""Tokenizers: a trainable word-level tokenizer for the BiLSTM and a thin HF wrapper for the transformer.

Both expose `encode(texts, max_len) -> dict(input_ids, attention_mask)` (torch tensors, dynamic padding).
"""
from __future__ import annotations

import json
from collections import Counter

import torch

from src.preprocessing import simple_tokenize

PAD, UNK = "<pad>", "<unk>"


class WordTokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.itos = vocab or [PAD, UNK]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2, max_size: int = 30000) -> "WordTokenizer":
        c = Counter(tok for t in texts for tok in simple_tokenize(t))
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        return cls([PAD, UNK] + words)

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        return simple_tokenize(text)

    def encode_one(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in simple_tokenize(text)[:max_len]]
        return ids or [1]

    def encode(self, texts: list[str], max_len: int, truncation: str = "head", head_tokens: int = 64) -> dict:
        seqs = [self.encode_one(t, max_len) for t in texts]  # word-level model: head truncation only
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def to_json(self) -> str:
        return json.dumps(self.itos, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "WordTokenizer":
        return cls(json.loads(s))


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.name = name_or_path
        self.tok = AutoTokenizer.from_pretrained(name_or_path)

    def tokenize(self, text: str) -> list[str]:
        return self.tok.tokenize(text)

    def _special_wrap(self) -> tuple[list[int], list[int]]:
        """Special-token prefix/suffix of a single sequence, read off the tokenizer's own output
        (e.g. [CLS] .. [SEP] for BERT/MuRIL, <s> .. </s> for RoBERTa/XLM-R)."""
        if not hasattr(self, "_wrap"):
            full = self.tok("a", add_special_tokens=True)["input_ids"]
            core = self.tok("a", add_special_tokens=False)["input_ids"]
            i = next(k for k in range(len(full)) if full[k : k + len(core)] == core)
            self._wrap = (full[:i], full[i + len(core):])
        return self._wrap

    def encode(self, texts: list[str], max_len: int, truncation: str = "head", head_tokens: int = 64) -> dict:
        """truncation="head": keep the first tokens (default, original behaviour).
        truncation="head_tail": keep the first `head_tokens` and the last (budget - head_tokens) tokens
        (Sun et al., 2019, Sec. 5.3.1)."""
        if truncation == "head":
            b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            return {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
        if truncation != "head_tail":
            raise ValueError(f"unknown truncation strategy: {truncation}")
        prefix, suffix = self._special_wrap()
        budget = max_len - len(prefix) - len(suffix)
        if not 0 < head_tokens < budget:
            raise ValueError(f"head_tokens must be in (0, {budget})")
        seqs = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            if len(ids) > budget:
                ids = ids[:head_tokens] + ids[len(ids) - (budget - head_tokens):]
            seqs.append(prefix + ids + suffix)
        L = max(len(x) for x in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask}
