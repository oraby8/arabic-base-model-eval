# Which open LLM is the best starting point for Egyptian, Saudi and Gulf Arabic?

A companion to [arabic-tokenizer-eval](../arabic-tokenizer-eval), which measured tokenizers. This project measures how well the models themselves already know each target dialect, to pick a base model for continued pretraining or fine-tuning. Each dialect is reported **on its own**: Egyptian, Saudi (Najdi and Hijazi) and Gulf/Emirati. MSA and English are references.

**Results: [RESULTS.md](RESULTS.md)**, plus `results/ranking.md`, `results/bpc_by_set.md` and `results/bpc_vs_size.png`.

## Metrics
Every model gets the same held-out text and the same code.

| Metric | What it tells you | Why it is fair across models |
|---|---|---|
| **Bits per character (BPC)** per dialect | How well the model predicts the dialect. This is the most direct measure of language knowledge, and it is what continued pretraining improves. | It is divided by the characters of the original text, so the tokenizer doesn't change it. |
| **FLORES gap** = dialect BPC / MSA BPC on the same FLORES+ sentences | The dialect penalty with the content held fixed (Egyptian and Saudi only) | It is a ratio within one model, on parallel text. |
| **Fertility** (tokens per word) | Training and inference cost per word | Measured on the same text for every model. |
| **Multiple choice:** Belebele for MSA, Egyptian, Najdi and English (900 questions each); Alyah for Emirati (1,173) | Zero-shot comprehension and knowledge | Each option is scored by its log-likelihood per character. |
| **Paired bootstrap 95% CI** of each model's difference to the best one | Whether a gap is real | The same document resamples are used for every model. |

## Evaluation data (`src/prep_eval.py`)
A dialect's BPC is the unweighted mean over its sets, so no single source dominates.

| Dialect | Sets |
|---|---|
| MSA | FLORES+ `arb_Arab`, PADIC MSA, FineWeb-2 `arb_Arab` (ALDi < 0.1) |
| Egyptian | FLORES+ `arz_Arab`, FineWeb-2 `arz_Arab` (ALDi ≥ 0.3), Omnilingual ASR `arz_Arab` (speech transcripts, 15 speakers) |
| Saudi | FLORES+ `ars_Arab` (Najdi), FineWeb-2 `ars_Arab`, Omnilingual `ars_Arab` (Najdi) and `acw_Arab` (Hijazi) |
| Gulf | Omnilingual `afb_Arab` (11 speakers), Emirati TV/podcast transcripts (`eabayed/EmiratiDialictShowsAudioTranscription`), Emirati radio (`FatimahEmadEldin/alsanaa-emirati-arabic-asr`) |
| English | FLORES+ `eng_Latn` |

- **Size:** 4,151 documents, 4.2M characters.
- **No truncation:** every document is at most 2,500 characters, so no model ever truncates.
- **Speech transcripts:** annotation tags are removed. Each speaker's utterances are joined into documents of about 1,000 characters, with at most 25 documents per speaker.
- **Gulf has no web or FLORES text,** so it is speech only. Compare models *within* a dialect, not dialects with each other.
- **Calibration text:** `data/calib.parquet` holds 120 extra FineWeb-2 documents (MSA, Egyptian, Najdi) that are not in the evaluation set. They are used only to choose each model's start token.

## Differences from a simple `diag.py`
The usual quick script (`labels=input_ids`, `chars = len(decode(...))`, truncation at 2,048 tokens) gives numbers that aren't comparable across models:
1. **Every character is scored.**
   - Each document is prefixed with a start token that is itself never scored. Without one, the first token is free while its characters still count.
   - The start token is the tokenizer's default, BOS, EOS or a newline, whichever gives the lowest BPC for that model on the calibration text.
   - It matters a lot. For example, Qwen3-8B-Base scores 1.98 BPC after `<|endoftext|>` but 1.62 after a newline. Gemma-3 scores 2.85 without BOS and 1.32 with it. Any single fixed rule would favour some models over others.
2. **The denominator is the original text**, not a decode. All 21 tokenizers decode the text back exactly and produce zero unknown tokens (`meta.json`).
3. **No truncation.** Every model scores every character of identical documents.
4. **Base checkpoints wherever they exist.** Models with only a chat checkpoint are marked `post`.
5. **Per-document bits and per-question answers are saved,** so the differences between models get confidence intervals.

## Models
There are 21 models, listed in `src/models.py`. Llama-4-Scout, DeepSeek-V3 and Command-A are excluded, because their weights exceed the server's free disk.

## Reproduce
```bash
cp env.sh.example env.sh && . ./env.sh    # B, TOK_EVAL (path to arabic-tokenizer-eval), HF_HOME
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python src/prep_eval.py                   # -> data/eval.parquet, data/calib.parquet
python src/run_all.py --disk-gb 95        # every model: download, evaluate, delete weights (4 GPUs)
python src/summarize.py                   # -> results/summary.csv, ranking.md, bpc_by_set.md, bpc_vs_size.png
# one model: CUDA_VISIBLE_DEVICES=0 python src/diag.py --model Qwen3-8B
```
- **Hardware:** 4× L40S (46 GB). The whole run takes about 5 hours, mostly downloading weights.
- **GPU memory:** a batch that runs out of GPU memory is split in half and retried.
- **Disk:** downloaded weights are deleted even when a job fails. `run_all.py` retries each failed job twice.

## Layout
```
src/models.py      model list (checkpoint, GPUs, download size, batch budget)
src/prep_eval.py   builds the held-out evaluation and calibration sets
src/diag.py        start-token calibration, BPC, fertility, Belebele/Alyah for one model
src/run_all.py     schedules all models on free GPUs within a disk budget
src/summarize.py   per-dialect rankings, paired bootstrap CIs, figure
results/models/<model>/  bpc.csv, docs.parquet (per document), mcq.csv, mcq_items.parquet, meta.json
```
