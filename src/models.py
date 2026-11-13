"""Models under evaluation: the LLMs from arabic-tokenizer-eval, using the base (pretrained-only)
checkpoint wherever one is public, since the question is which model is the best starting point.

stage: "base" = pretrained only; "post" = only an instruction-tuned/chat checkpoint is public
       (its BPC on raw text is usually somewhat worse than its base would be; read it with that in mind).
gpus:  GPUs needed in bf16 on 46 GB cards (L40S), with weights capped at 36 GiB per card. size_gb: download size (disk budget).
active_b: active parameters for MoE models (billions).
remote: needs trust_remote_code (custom modelling code from the repo).

Not evaluated (weights larger than the server's free disk/VRAM):
  Llama-4-Scout-17B-16E (217 GB), DeepSeek-V3 (~690 GB), Command-A (222 GB).
"""

MODELS = [
    # --- global ---
    dict(name="Llama-3.2-1B", repo="meta-llama/Llama-3.2-1B", stage="base", group="global", gpus=1, size_gb=2.5),
    dict(name="Llama-3.1-8B", repo="meta-llama/Llama-3.1-8B", stage="base", group="global", gpus=1, size_gb=16.1),
    dict(name="Qwen2.5-7B", repo="Qwen/Qwen2.5-7B", stage="base", group="global", gpus=1, size_gb=15.2),
    dict(name="Qwen3-8B", repo="Qwen/Qwen3-8B-Base", stage="base", group="global", gpus=1, size_gb=16.4),
    dict(name="Qwen3.5-9B", repo="Qwen/Qwen3.5-9B-Base", stage="base", group="global", gpus=1, size_gb=19.3),
    dict(name="Gemma-3-12B", repo="google/gemma-3-12b-pt", stage="base", group="global", gpus=1, size_gb=24.4),
    dict(name="Gemma-4-31B", repo="google/gemma-4-31B", stage="base", group="global", gpus=2, size_gb=62.5),
    dict(name="Falcon-H1-7B", repo="tiiuae/Falcon-H1-7B-Base", stage="base", group="global", gpus=1, size_gb=15.2),
    dict(name="Aya-Expanse-8B", repo="CohereLabs/aya-expanse-8b", stage="post", group="global", gpus=1, size_gb=16.1),
    dict(name="Command-R-32B", repo="CohereLabs/c4ai-command-r-08-2024", stage="post", group="global", gpus=2, size_gb=64.6),
    dict(name="gpt-oss-20B", repo="openai/gpt-oss-20b", stage="post", group="global", gpus=2, size_gb=13.8, active_b=3.6),
    # --- Arabic-centric ---
    dict(name="Command-R7B-Arabic", repo="CohereLabs/c4ai-command-r7b-arabic-02-2025", stage="post", group="arabic", gpus=1, size_gb=16.1),
    dict(name="Jais-3-0.9B", repo="inception42/Jais-3-0.9B", stage="base", group="arabic", gpus=1, size_gb=2.2, remote=True),
    dict(name="Jais-adapted-7B", repo="inception42/jais-adapted-7b", stage="base", group="arabic", gpus=1, size_gb=28.0),
    dict(name="ALLaM-7B", repo="humain-ai/ALLaM-7B-Instruct-preview", stage="post", group="arabic", gpus=1, size_gb=14.0),
    dict(name="Fanar-1-9B", repo="QCRI/Fanar-1-9B", stage="base", group="arabic", gpus=1, size_gb=18.5),
    dict(name="Fanar-2-27B", repo="QCRI/Fanar-2-27B-Instruct", stage="post", group="arabic", gpus=2, size_gb=54.0),
    dict(name="Karnak-6B", repo="Applied-Innovation-Center/Karnak-6B-v1.0", stage="base", group="arabic", gpus=1, size_gb=11.9),
    dict(name="Karnak-40B", repo="Applied-Innovation-Center/Karnak-40B-v1.0", stage="base", group="arabic", gpus=3, size_gb=81.3),
    dict(name="Atlas-Chat-9B", repo="MBZUAI-Paris/Atlas-Chat-9B", stage="post", group="arabic", gpus=1, size_gb=18.5),
    dict(name="Nile-Chat-12B", repo="MBZUAI-Paris/Nile-Chat-12B", stage="post", group="arabic", gpus=1, size_gb=23.5),
]
BY_NAME = {m["name"]: m for m in MODELS}

# Belebele varieties (FLORES+ codes): MSA, Egyptian, Najdi (Saudi), English. Emirati/Gulf: Alyah (in diag.py).
BELEBELE = ["arb_Arab", "arz_Arab", "ars_Arab", "eng_Latn"]
DIALECTS = ["Egyptian", "Saudi", "Gulf"]          # target dialects, each reported on its own
REFERENCES = ["MSA", "English"]
