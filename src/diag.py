"""Diagnose one candidate base model on held-out MSA and dialect text.

Metrics (all on identical text for every model, data/eval.parquet):
  BPC   bits per character = total -log2 p(text) / characters. The text is prefixed with a start token so
        *every* character is predicted. The start token (tokenizer default, BOS, EOS or a newline) is chosen per
        model on held-out calibration text (data/calib.parquet), since the best one differs between models,
        and the denominator is the length of the original text, not of a decode. Tokenizer-independent.
  fertility  tokens per whitespace word, and chars/token (no special tokens).
  Belebele  zero-shot, 900 questions per variety (MSA, Egyptian, Najdi, English); Alyah: 1,173 Emirati questions. Context "P: {passage}\nQ: {question}\nA:" then " {option}";
        score = log p(option | context) / characters of option, so options compete fairly across tokenizers.
Fairness checks saved in meta.json: decode round-trip rate and unknown-token rate (a lossy tokenizer
gets an unfair BPC advantage), and that no document was truncated.
Per-document bits (docs.parquet) and per-question correctness (mcq_items.parquet) are saved, so differences between models can be paired.

Usage: CUDA_VISIBLE_DEVICES=0 python src/diag.py --model Qwen3-8B [--max-docs 20] [--keep-weights]
"""
import argparse, atexit, glob, json, math, os, shutil, time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from models import BY_NAME, BELEBELE

B = os.environ["B"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max-docs", type=int, default=0, help="smoke test: this many docs per set / questions per language")
ap.add_argument("--tok-budget", type=int, default=12288, help="max padded tokens per forward batch")
ap.add_argument("--keep-weights", action="store_true", help="do not delete the downloaded weights afterwards")
a = ap.parse_args()
cfg = BY_NAME[a.model]
out = f"{B}/results/models/{a.model}" + ("-smoke" if a.max_docs else "")
os.makedirs(out, exist_ok=True)
t0 = time.time()

# ---------- load ----------
files = [s.rfilename for s in HfApi().model_info(cfg["repo"]).siblings]
ignore = ["original/*", "metal/*", "*.gguf", "*.pth", "consolidated*"]
if any(f.endswith(".safetensors") for f in files):
    ignore.append("*.bin")
from huggingface_hub.constants import HF_HUB_CACHE
repo_dir = os.path.join(HF_HUB_CACHE, "models--" + cfg["repo"].replace("/", "--"))


def drop_weights():
    """Delete this model's files, including blobs in the (shared) blob store and partial downloads.
    Runs at exit even when the job fails, so a failed job never leaves tens of GB behind."""
    if a.keep_weights:
        return
    for root, _, fs in os.walk(repo_dir):
        for f in fs:
            target = os.path.realpath(os.path.join(root, f))
            if target != os.path.join(root, f) and os.path.exists(target):
                os.remove(target)
    shutil.rmtree(repo_dir, ignore_errors=True)
    for f in glob.glob(os.path.join(HF_HUB_CACHE, "blobs", "*.incomplete")) + glob.glob(os.path.join(repo_dir, "blobs", "*.incomplete")):
        os.remove(f)


atexit.register(drop_weights)
path = snapshot_download(cfg["repo"], ignore_patterns=ignore)
from transformers import AutoModelForCausalLM, AutoTokenizer
remote = cfg.get("remote", False)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=remote)
# leave ~8 GB per card for activations and the [batch, length, vocab] logits (vocabularies reach 262K)
max_memory = {i: "36GiB" for i in range(torch.cuda.device_count())}
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory,
                                             trust_remote_code=remote).eval()
dev = next(model.parameters()).device
n_params = sum(p.numel() for p in model.parameters())
t_load = time.time() - t0  # download + load
t0 = time.time()

# The start token this model was trained to see: whatever its tokenizer prepends by default.
probe, bare = tok("سلام").input_ids, tok("سلام", add_special_tokens=False).input_ids
default = probe[:len(probe) - len(bare)] if probe[-len(bare):] == bare else []
PAD = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)


def enc(texts):
    return tok(texts, add_special_tokens=False).input_ids


def logprobs(seqs):
    """list of id lists -> list of float arrays: log p(seq[t] | seq[<t]) for t >= 1 (natural log).
    On CUDA OOM the batch is split in half and retried (results are identical, only slower)."""
    try:
        return _logprobs(seqs)
    except torch.OutOfMemoryError:
        if len(seqs) == 1:
            raise
        torch.cuda.empty_cache()
        h = len(seqs) // 2
        return logprobs(seqs[:h]) + logprobs(seqs[h:])


@torch.no_grad()
def _logprobs(seqs):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    x, m = x.to(dev), m.to(dev)
    logits = model(input_ids=x, attention_mask=m).logits
    res = []
    for i, s in enumerate(seqs):  # row by row: never materialise float32 [B, L, V]
        n = len(s) - 1
        lp = -torch.nn.functional.cross_entropy(logits[i, :n].float(), x[i, 1:n + 1], reduction="none")
        res.append(lp.cpu().numpy())
    del logits
    return res


def batches(lengths):
    """index batches over length-sorted items, each <= tok_budget padded tokens."""
    order = np.argsort(lengths)[::-1]
    cur, mx = [], 0
    for i in order:
        if cur and max(mx, lengths[i]) * (len(cur) + 1) > a.tok_budget:
            yield cur; cur, mx = [], 0
        cur.append(i); mx = max(mx, lengths[i])
    if cur:
        yield cur


# ---------- start token, chosen on held-out calibration text ----------
# The best conditioning token differs by model (Qwen3-Base, for instance, does much worse after <|endoftext|>
# than after a newline). Each model gets the candidate that minimises its BPC on data/calib.parquet
# (FineWeb-2 docs not in the evaluation set); the start token itself is never scored.
cands = {"default": default, "bos": [tok.bos_token_id] if tok.bos_token_id is not None else [],
         "eos": [tok.eos_token_id] if tok.eos_token_id is not None else [], "newline": enc(["\n"])[0]}
cal = pd.read_parquet(f"{B}/data/calib.parquet").text.tolist()
cal_ids, cal_chars, cal_bpc = enc(cal), sum(len(t) for t in cal), {}
for name, pre in cands.items():
    if not pre or pre in [cands[k] for k in cal_bpc]:
        continue
    seqs = [pre + s for s in cal_ids]
    bits = 0.0
    for bi in batches(np.array([len(s) for s in seqs])):
        bits += sum(-lp[len(pre) - 1:].sum() for lp in logprobs([seqs[i] for i in bi])) / math.log(2)
    cal_bpc[name] = float(bits / cal_chars)
start = min(cal_bpc, key=cal_bpc.get)
prefix = cands[start]
print("start-token calibration (BPC):", {k: round(v, 4) for k, v in cal_bpc.items()}, "->", start, prefix, flush=True)

# ---------- BPC + fertility ----------
df = pd.read_parquet(f"{B}/data/eval.parquet")
if a.max_docs:
    df = df.groupby("set").head(a.max_docs).reset_index(drop=True)
ids = enc(df.text.tolist())
seqs = [prefix + s for s in ids]
ctx = getattr(model.config, "max_position_embeddings", None) or getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 10**9)
assert max(len(s) for s in seqs) <= ctx, "a document exceeds the model context"
bits = np.zeros(len(df))
for bi in batches(np.array([len(s) for s in seqs])):
    for i, lp in zip(bi, logprobs([seqs[i] for i in bi])):
        bits[i] = -lp[len(prefix) - 1:].sum() / math.log(2)  # only the text's own tokens
df["bits"], df["chars"] = bits, df.text.str.len()
df["tokens"], df["words"] = [len(s) for s in ids], df.text.str.split().str.len()
unk = tok.unk_token_id
df["unk"] = [sum(t == unk for t in s) if unk is not None else 0 for s in ids]
df["roundtrip"] = [tok.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=False) == t for s, t in zip(ids, df.text)]
df[["dialect", "set", "doc", "bits", "chars", "tokens", "words", "unk", "roundtrip"]].to_parquet(f"{out}/docs.parquet", index=False)
g = df.groupby(["dialect", "set"]).agg(bits=("bits", "sum"), chars=("chars", "sum"), tokens=("tokens", "sum"),
                          words=("words", "sum"), docs=("doc", "size"), unk=("unk", "sum"), roundtrip=("roundtrip", "mean"))
g["bpc"], g["fertility"], g["chars_per_token"] = g.bits / g.chars, g.tokens / g.words, g.chars / g.tokens
g.reset_index().to_csv(f"{out}/bpc.csv", index=False)
print(a.model, "BPC\n", g[["bpc", "fertility", "roundtrip"]].round(3).to_string(), flush=True)
t_bpc = time.time() - t0

# ---------- multiple choice: Belebele (MSA, Egyptian, Najdi, English) + Alyah (Emirati) ----------
def mcq():
    """yields (bench, lang, qid, context, [4 options], answer index)"""
    for lang in BELEBELE:
        qs = [json.loads(l) for l in open(hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset"))]
        for q in qs[:a.max_docs or None]:
            yield ("belebele", lang, f"{q['link']}|{q['question_number']}", f"P: {q['flores_passage']}\nQ: {q['question']}\nA:",
                   [q[f"mc_answer{j}"] for j in range(1, 5)], int(q["correct_answer_num"]) - 1)
    al = pd.read_parquet(hf_hub_download("tiiuae/alyah-emirati-benchmark", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    for k, q in al.head(a.max_docs or None).iterrows():
        yield ("alyah", "emirati", f"{k}|{q.category}", f"Q: {q.query}\nA:",
               [q[f"option_{j}"] for j in range(1, 5)], int(q.correct_answer) - 1)


items, seqs, meta = [], [], []
for bench, lang, qid, ctx, options, ans in mcq():
    c = enc([ctx])[0]
    opts = [" " + str(o).strip() for o in options]
    for j, o in enumerate(enc(opts)):
        seqs.append(prefix + c + o); meta.append((len(items), j, len(prefix) + len(c) - 1, len(opts[j].strip())))
    items.append(dict(bench=bench, lang=lang, qid=qid, answer=ans))
score = np.full((len(items), 4), -np.inf)
for bi in batches(np.array([len(s) for s in seqs])):
    for i, lp in zip(bi, logprobs([seqs[i] for i in bi])):
        it, j, start, nch = meta[i]
        score[it, j] = lp[start:].sum() / max(nch, 1)  # log-likelihood per character of the option
mc = pd.DataFrame(items)
mc["pred"] = score.argmax(1)
mc["correct"] = mc.pred == mc.answer
mc.to_parquet(f"{out}/mcq_items.parquet", index=False)
acc = mc.groupby(["bench", "lang"]).correct.agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"}).reset_index()
acc["ci95"] = 1.96 * np.sqrt(acc.acc * (1 - acc.acc) / acc.n)
acc.to_csv(f"{out}/mcq.csv", index=False)
print(acc.round(3).to_string(index=False), flush=True)

json.dump(dict(model=a.model, repo=cfg["repo"], stage=cfg["stage"], group=cfg["group"], n_params=n_params,
               active_b=cfg.get("active_b"), prefix=prefix, start=start, start_calibration=cal_bpc, vocab=len(tok), docs=len(df),
               roundtrip_rate=float(df.roundtrip.mean()), unk_tokens=int(df.unk.sum()),
               max_doc_tokens=int(max(len(s) for s in ids)) + len(prefix),
               gpus=torch.cuda.device_count(), minutes_load=round(t_load / 60, 1), minutes_bpc=round(t_bpc / 60, 1),
               minutes_total=round((time.time() - t0) / 60, 1)), open(f"{out}/meta.json", "w"), indent=1)
print("done", a.model, f"{(time.time() - t0) / 60:.1f} min", flush=True)
