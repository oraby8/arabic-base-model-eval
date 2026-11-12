"""Build the held-out evaluation text: data/eval.parquet (set, source, variety, doc, text).

Source text is arabic-tokenizer-eval's prepared data (raw text, NFC, diacritics kept as written):
  FLORES+ dev+devtest (English, MSA, 9 dialects): parallel sentences. Every 8 consecutive sentence ids form one
      document, joined with spaces, so each document has the same content in every variety.
  PADIC (MSA + 4 dialects): parallel sentences. Every 30 consecutive ids form one document, joined with newlines.
  FineWeb-2 test split, filtered with ALDi as in arabic-tokenizer-eval (MSA < 0.1, dialects >= 0.3).
      Not parallel. Documents are shuffled (seed 0), split at whitespace into pieces of at most PIECE characters,
      and kept until a variety reaches CAP characters.
Every document is at most MAXC = 2,500 characters, so it fits in every model's context (>= 4,096 tokens) even at
1 char/token: no model ever truncates.
Usage: python src/prep_eval.py        (reads $TOK_EVAL, writes $B/data/eval.parquet)
"""
import os, unicodedata
import numpy as np
import pandas as pd

TOK, B = os.environ["TOK_EVAL"], os.environ["B"]
MAXC, PIECE, CAP = 2500, 1500, 400_000


def nfc(s):
    return unicodedata.normalize("NFC", s).strip()


def sid_key(s):
    """'dev-12' -> ('dev', 12): numeric order keeps FLORES sentences from the same article together."""
    head, _, num = str(s).rpartition("-")
    return (head, int(num)) if num.isdigit() else (str(s), 0)


def parallel(df, source, group, sep):
    out = []
    for v, g in df.groupby("variety"):
        g = g.iloc[sorted(range(len(g)), key=lambda i: sid_key(g.sid.iloc[i]))].reset_index(drop=True)
        for k in range(0, len(g), group):
            out.append((f"{source}:{v}", source, v, f"{source}:{k // group}", sep.join(nfc(t) for t in g.text_raw[k:k + group])))
    return out


def split(text, maxc=PIECE):
    words, cur = text.split(" "), []
    pieces, n = [], 0
    for w in words:
        if n + len(w) + 1 > maxc and cur:
            pieces.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += len(w) + 1
    if cur:
        pieces.append(" ".join(cur))
    return [p.strip() for p in pieces if len(p.strip()) >= 50 and len(p) <= maxc]


rows = []
P = f"{TOK}/data/prepared"
rows += parallel(pd.read_parquet(f"{P}/flores.parquet", columns=["sid", "variety", "text_raw"]), "flores", 8, " ")
rows += parallel(pd.read_parquet(f"{P}/padic.parquet", columns=["sid", "variety", "text_raw"]), "padic", 30, "\n")
w = pd.read_parquet(f"{P}/fineweb.parquet", columns=["sid", "variety", "text_raw"]).merge(
    pd.read_parquet(f"{TOK}/results/fineweb/aldi.parquet"), on=["sid", "variety"])
w = w[((w.variety == "arb_Arab") & (w.aldi < 0.1)) | ((w.variety != "arb_Arab") & (w.aldi >= 0.3))]
rng = np.random.default_rng(0)
for v, g in w.groupby("variety"):
    total = 0
    for i in rng.permutation(len(g)):
        for j, piece in enumerate(split(nfc(g.text_raw.iloc[i]))):
            rows.append((f"fineweb:{v}", "fineweb", v, f"fineweb:{g.sid.iloc[i]}:{j}", piece)); total += len(piece)
        if total >= CAP:
            break

df = pd.DataFrame(rows, columns=["set", "source", "variety", "doc", "text"])
assert df.text.str.len().max() <= MAXC, df.text.str.len().max()
os.makedirs(f"{B}/data", exist_ok=True)
df.to_parquet(f"{B}/data/eval.parquet", index=False)
s = df.assign(chars=df.text.str.len()).groupby("set").agg(docs=("doc", "size"), chars=("chars", "sum"))
print(s.to_string()); print("total", len(df), "docs,", s.chars.sum(), "chars")
