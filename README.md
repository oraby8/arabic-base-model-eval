# Which open LLM is the best starting point for Arabic?

A companion to [arabic-tokenizer-eval](../arabic-tokenizer-eval). That project measured tokenizers. This one measures how well the models themselves already know MSA and dialectal Arabic, to pick a base model for continued pretraining or fine-tuning.

Each model is scored on the same held-out text, with the same code:

| Metric | What it tells you | Why it is fair across models |
|---|---|---|
| **Bits per character (BPC)** on MSA and dialects | How well the model predicts the text. This is the most direct measure of language knowledge, and it is what continued pretraining improves. | It is divided by characters, not tokens, so a tokenizer that splits text into more pieces doesn't look better or worse. |
| **Dialect gap** = dialect BPC / MSA BPC | How much worse the model is on dialects than on MSA | It is a ratio within one model. |
| **Arabic tax** = MSA BPC / English BPC | How much worse the model is on Arabic than on English | It uses FLORES+, the same content in both languages. |
| **Fertility** (tokens per word) and chars/token | Training and inference cost per word of Arabic | Measured on the same text for every model. |
| **Belebele** (zero-shot, 900 questions per variety) | Reading comprehension in MSA, 5 dialects and English | Each answer option is scored by its log-likelihood per character. |

## Evaluation data
All of it is held out from arabic-tokenizer-eval's training data. `src/prep_eval.py` builds it:
- **FLORES+** dev and devtest, in English, MSA and 9 dialects. Every 8 consecutive sentences form one document, so each document has the same content in every variety.
- **PADIC:** MSA plus 2 Algerian, Palestinian and Syrian. These are parallel sentences, grouped 30 to a document.
- **FineWeb-2 test split:** web text, filtered with ALDi to MSA < 0.1 and dialects ≥ 0.3. Documents are split into pieces of at most 1,500 characters, about 400K characters per variety. Moroccan has only 89K characters left after filtering.

In total: 6,826 documents, 6.2M characters and 23 sets. Every document is at most 2,500 characters, so no model ever truncates.

## Differences from a simple `diag.py`
The usual quick script (`labels=input_ids`, `chars = len(decode(...))`, truncation at 2,048 tokens) gives numbers that aren't comparable across models:
1. **The first token is never scored.** Its characters still count in the denominator. Without an explicit start token, models with longer first tokens look better. Here the text is prefixed with the model's own start token (BOS, else EOS), so every character is predicted.
2. **The denominator is the original text**, not a decode, which some tokenizers normalize. The decode round-trip rate and the count of unknown tokens are reported in `meta.json`. A lossy tokenizer would otherwise get a free BPC advantage.
3. **No truncation.** With a fixed token limit, models with less efficient tokenizers are scored on less text, and a different part of it. Here every model scores every character of identical documents.
4. **Base checkpoints wherever they exist.** For example `Qwen3-8B-Base` rather than `Qwen3-8B`, and `gemma-4-31B` rather than `-it`. Models with only a chat checkpoint are marked `post`.
5. **Per-document bits and per-question Belebele results are saved,** so the differences between models get paired bootstrap confidence intervals.

## Models
There are 21 models, listed in `src/models.py`. They are the LLMs from arabic-tokenizer-eval, minus three whose weights exceed the server's disk: Llama-4-Scout (217 GB), DeepSeek-V3 and Command-A.

## Reproduce
```bash
cp env.sh.example env.sh && . ./env.sh    # B, TOK_EVAL (path to arabic-tokenizer-eval), HF_HOME
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python src/prep_eval.py                   # -> data/eval.parquet
python src/run_all.py --disk-gb 95        # every model: download, evaluate, delete weights (4 GPUs)
python src/summarize.py                   # -> results/summary.csv, ranking.md, bpc_vs_size.png
# one model: CUDA_VISIBLE_DEVICES=0 python src/diag.py --model Qwen3-8B
```
Hardware: 4× L40S (46 GB). Models of 27B and larger are split across 2 GPUs.

## Layout
```
src/models.py      model list (base checkpoint, GPUs, download size)
src/prep_eval.py   builds the held-out evaluation set
src/diag.py        BPC, fertility, Belebele for one model
src/run_all.py     schedules all models on free GPUs within a disk budget
src/summarize.py   ranking table, paired bootstrap CIs, figure
results/models/<model>/  bpc.csv, docs.parquet (per document), belebele.csv, belebele_items.parquet, meta.json
```
