"""Build the held-out evaluation text: data/eval.parquet (set, source, variety, dialect, doc, text).

Target varieties: Egyptian, Saudi, Gulf/Emirati, with MSA and English as references.

| dialect   | set                    | what it is |
|-----------|------------------------|------------|
| MSA       | flores:arb_Arab        | FLORES+ dev+devtest (Wikipedia-style news), 8 sentences per doc |
|           | padic:padic_msa        | PADIC MSA, 30 sentences per doc |
|           | fineweb:arb_Arab       | FineWeb-2 test, ALDi < 0.1 |
| Egyptian  | flores:arz_Arab        | FLORES+ Egyptian (same content as flores:arb_Arab) |
|           | fineweb:arz_Arab       | FineWeb-2 test, ALDi >= 0.3 |
|           | omni:arz_Arab          | Omnilingual ASR corpus (Meta, CC-BY-4.0): transcribed spontaneous speech |
| Saudi     | flores:ars_Arab        | FLORES+ Najdi |
|           | fineweb:ars_Arab       | FineWeb-2 test Najdi, ALDi >= 0.3 |
|           | omni:ars_Arab          | Omnilingual, Najdi speech |
|           | omni:acw_Arab          | Omnilingual, Hijazi speech |
| Gulf      | omni:afb_Arab          | Omnilingual, Gulf Arabic speech |
|           | emirati:shows          | eabayed/EmiratiDialictShowsAudioTranscription: Emirati TV, podcasts, YouTube |
|           | emirati:alsanaa        | FatimahEmadEldin/alsanaa-emirati-arabic-asr: traditional Emirati radio (one book) |
| English   | flores:eng_Latn        | FLORES+ English (same content as flores:arb_Arab) |

Speech transcripts: <<...>> annotation tags removed; utterances of one speaker joined into docs of ~1,000 chars;
Omnilingual: all shards, at most 25 docs per speaker.
Raw text, NFC, diacritics as written. Every doc <= 2,500 chars, so no model ever truncates.
Usage: python src/prep_eval.py        (reads $TOK_EVAL for FLORES/PADIC/FineWeb, downloads the rest; writes $B/data/eval.parquet)
"""
import os, re, unicodedata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

TOK, B = os.environ["TOK_EVAL"], os.environ["B"]
MAXC, PIECE, CAP, SPEECH_DOC = 2500, 1500, 400_000, 1000
DIALECT = {"arb_Arab": "MSA", "padic_msa": "MSA", "arz_Arab": "Egyptian", "ars_Arab": "Saudi", "acw_Arab": "Saudi",
           "afb_Arab": "Gulf", "shows": "Gulf", "alsanaa": "Gulf", "eng_Latn": "English"}
OMNI = ["arz_Arab", "ars_Arab", "acw_Arab", "afb_Arab"]   # every shard is read: each holds only 2-3 speakers
PER_SPEAKER = 25                                        # max docs (~25K chars) per speaker, so no speaker dominates


def nfc(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def sid_key(s):
    """'dev-12' -> ('dev', 12): numeric order keeps FLORES sentences from the same article together."""
    head, _, num = str(s).rpartition("-")
    return (head, int(num)) if num.isdigit() else (str(s), 0)


def parallel(df, source, group, sep):
    out = []
    for v, g in df.groupby("variety"):
        g = g.iloc[sorted(range(len(g)), key=lambda i: sid_key(g.sid.iloc[i]))].reset_index(drop=True)
        for k in range(0, len(g), group):
            out.append((f"{source}:{v}", source, v, f"{source}:{v}:{k // group}", sep.join(nfc(t) for t in g.text_raw[k:k + group])))
    return out


def split(text, maxc=PIECE):
    pieces, cur, n = [], [], 0
    for w in text.split(" "):
        if n + len(w) + 1 > maxc and cur:
            pieces.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += len(w) + 1
    if cur:
        pieces.append(" ".join(cur))
    return [p.strip() for p in pieces if len(p.strip()) >= 50 and len(p) <= maxc]


def speech_docs(utts, source, variety, cap=CAP, per_speaker=10**9):
    """utts: DataFrame(speaker, text). Join one speaker's utterances (in order) into ~SPEECH_DOC-char docs."""
    out, total = [], 0
    for spk, g in utts.groupby("speaker", sort=True):
        cur, n, k = [], 0, 0
        for t in g.text:
            if k >= per_speaker:
                break
            cur.append(t); n += len(t) + 1
            if n >= SPEECH_DOC:
                out.append((f"{source}:{variety}", source, variety, f"{source}:{variety}:{spk}:{k}", " ".join(cur))); total += n
                cur, n, k = [], 0, k + 1
        if cur and n >= 100 and k < per_speaker:
            out.append((f"{source}:{variety}", source, variety, f"{source}:{variety}:{spk}:{k}", " ".join(cur))); total += n
        if total >= cap:
            break
    return out


def clean_speech(t):
    t = re.sub(r"<<[^>]*>\s*>|<[^>]*>|\[[^\]]*\]", " ", str(t))
    return nfc(t)


rows = []
P = f"{TOK}/data/prepared"
keep = {"arb_Arab", "arz_Arab", "ars_Arab", "eng_Latn"}
f = pd.read_parquet(f"{P}/flores.parquet", columns=["sid", "variety", "text_raw"])
rows += parallel(f[f.variety.isin(keep)], "flores", 8, " ")
p = pd.read_parquet(f"{P}/padic.parquet", columns=["sid", "variety", "text_raw"])
rows += parallel(p[p.variety == "padic_msa"], "padic", 30, "\n")
w = pd.read_parquet(f"{P}/fineweb.parquet", columns=["sid", "variety", "text_raw"]).merge(
    pd.read_parquet(f"{TOK}/results/fineweb/aldi.parquet"), on=["sid", "variety"])
w = w[((w.variety == "arb_Arab") & (w.aldi < 0.1)) | (w.variety.isin(["arz_Arab", "ars_Arab"]) & (w.aldi >= 0.3))]
rng = np.random.default_rng(0)
for v, g in w.groupby("variety"):
    total = 0
    for i in rng.permutation(len(g)):
        for j, piece in enumerate(split(nfc(g.text_raw.iloc[i]))):
            rows.append((f"fineweb:{v}", "fineweb", v, f"fineweb:{v}:{g.sid.iloc[i]}:{j}", piece)); total += len(piece)
        if total >= CAP:
            break

# Omnilingual ASR corpus transcripts (text columns only; the audio is not read)
from huggingface_hub import HfApi
omni_files = HfApi().list_repo_files("facebook/omnilingual-asr-corpus", repo_type="dataset")
for v in OMNI:
    t = pd.concat([pq.read_table(hf_hub_download("facebook/omnilingual-asr-corpus", fn, repo_type="dataset"),
                                 columns=["speaker_id", "segment_id", "raw_text"]).to_pandas()
                   for fn in sorted(f for f in omni_files if f.startswith(f"data/{v}/") and f.endswith(".parquet"))])
    t = t.sort_values(["speaker_id", "segment_id"])
    t = pd.DataFrame(dict(speaker=t.speaker_id.astype(str), text=t.raw_text.map(clean_speech)))
    t = t[t.text.str.len() >= 10].drop_duplicates("text")
    rows += speech_docs(t, "omni", v, per_speaker=PER_SPEAKER)

# Emirati transcripts
e = pd.read_csv(hf_hub_download("eabayed/EmiratiDialictShowsAudioTranscription", "transcriptions.tsv", repo_type="dataset"), sep="\t")
e = pd.DataFrame(dict(speaker=e.audiofilename.str.extract(r"^([a-z]+)")[0].fillna("x"), text=e.transcription.map(clean_speech)))
rows += speech_docs(e[e.text.str.len() >= 10], "emirati", "shows")
a = pq.read_table(hf_hub_download("FatimahEmadEldin/alsanaa-emirati-arabic-asr", "data/train-00000-of-00001.parquet", repo_type="dataset"),
                  columns=["id", "transcription"]).to_pandas().sort_values("id")
for k, t in enumerate(a.transcription.map(clean_speech)):
    rows += [("emirati:alsanaa", "emirati", "alsanaa", f"emirati:alsanaa:{k}:{j}", piece) for j, piece in enumerate(split(t))]

df = pd.DataFrame(rows, columns=["set", "source", "variety", "doc", "text"])
df["dialect"] = df.variety.map(DIALECT)
assert df.dialect.notna().all() and df.text.str.len().max() <= MAXC, df.text.str.len().max()
assert df.doc.is_unique
os.makedirs(f"{B}/data", exist_ok=True)
df.to_parquet(f"{B}/data/eval.parquet", index=False)
s = df.assign(chars=df.text.str.len()).groupby(["dialect", "set"]).agg(docs=("doc", "size"), chars=("chars", "sum"))
print(s.to_string()); print("total", len(df), "docs,", s.chars.sum(), "chars")
