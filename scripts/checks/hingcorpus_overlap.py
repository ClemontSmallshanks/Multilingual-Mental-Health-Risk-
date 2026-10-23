"""Exact (normalised) overlap between HingCorpus and our hi-Latn sentiment val/test texts. Prints counts + ids only."""
import json, re, sys, tarfile, unicodedata
import pandas as pd
tar_path, processed, out = sys.argv[1], sys.argv[2], sys.argv[3]
def key(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"https?://\S+|www\.\S+|<url>|<email>|<phone>|@\w+|#", " ", t)
    t = "".join(c if (c.isalnum() or c.isspace()) else " " for c in t)
    return " ".join(t.split())
df = pd.read_csv(processed, keep_default_na=False)
df = df[(df.language == "hi-Latn") & df.split.isin(["val", "test"])]
df = df.assign(k=df.text.map(key))
df = df[df.k.str.split().str.len() >= 5]  # very short texts collide by chance
targets = {}
for r in df.itertuples():
    targets.setdefault(r.k, []).append((r.text_id, r.split, r.source))
hits, n_lines = set(), 0
with tarfile.open(tar_path, "r|gz") as tf:
    for m in tf:
        if not m.isfile():
            continue
        f = tf.extractfile(m)
        for raw in f:
            n_lines += 1
            k = key(raw.decode("utf-8", "ignore"))
            if k in targets:
                hits.add(k)
res = {"corpus_lines_scanned": n_lines, "checked_texts": len(df), "checked_by_split_source": df.groupby(["split", "source"]).size().astype(int).to_dict(),
       "exact_normalised_matches": sum(len(targets[k]) for k in hits),
       "matched_by_split_source": pd.Series([f"{s}|{src}" for k in hits for _, s, src in targets[k]]).value_counts().to_dict(),
       "matched_text_ids": sorted(i for k in hits for i, _, _ in targets[k])}
res["checked_by_split_source"] = {f"{a}|{b}": v for (a, b), v in res["checked_by_split_source"].items()}
json.dump(res, open(out, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "matched_text_ids"}, indent=2))
