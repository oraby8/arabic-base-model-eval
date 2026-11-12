"""Summarise every finished model, one dialect at a time (Egyptian, Saudi, Gulf), with MSA and English as references.

Outputs (results/):
  summary.csv        one row per model, every metric
  ranking.md         one ranking table per dialect + an overview table
  bpc_by_set.md      BPC on every evaluation set
  bpc_vs_size.png    dialect BPC vs model size, one panel per dialect

Per dialect: BPC = unweighted mean over that dialect's sets (see prep_eval.py), so no source dominates.
FLORES gap = FLORES dialect BPC / FLORES MSA BPC: the same sentences, so a clean measure of the dialect penalty
(Egyptian and Saudi only; FLORES+ has no Gulf variety).  Arabic tax = FLORES MSA BPC / FLORES English BPC.
CI: paired bootstrap over documents (1,000 resamples, identical for every model) of the difference to the best model.
Usage: B=<root> python src/summarize.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.environ.get("B", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{B}/src")
from models import MODELS, DIALECTS

R = f"{B}/results/models"
done = [m for m in MODELS if os.path.exists(f"{R}/{m['name']}/meta.json")]
ARABIC = ["MSA"] + DIALECTS
MCQ = {"MSA": ("belebele", "arb_Arab"), "Egyptian": ("belebele", "arz_Arab"), "Saudi": ("belebele", "ars_Arab"),
       "Gulf": ("alyah", "emirati"), "English": ("belebele", "eng_Latn")}
MCQ_NAME = {"MSA": "Belebele", "Egyptian": "Belebele", "Saudi": "Belebele (Najdi)", "Gulf": "Alyah (Emirati)", "English": "Belebele"}

rows, docs, sets_of = [], {}, {}
for m in done:
    n = m["name"]
    meta = json.load(open(f"{R}/{n}/meta.json"))
    b = pd.read_csv(f"{R}/{n}/bpc.csv")
    mc = pd.read_csv(f"{R}/{n}/mcq.csv").set_index(["bench", "lang"])
    docs[n] = pd.read_parquet(f"{R}/{n}/docs.parquet")
    sets_of = b.groupby("dialect").set.apply(list).to_dict()
    bs = b.set_index("set")
    rec = dict(model=n, group=m["group"], stage=m["stage"], params_b=meta["n_params"] / 1e9, active_b=m.get("active_b"))
    for d in ARABIC + ["English"]:
        ss = sets_of[d]
        rec[f"bpc_{d}"] = bs.bpc[ss].mean()
        rec[f"fert_{d}"] = bs.tokens[ss].sum() / bs.words[ss].sum()
        rec[f"mcq_{d}"] = 100 * mc.acc[MCQ[d]]
    for s in bs.index:
        rec[f"set:{s}"] = bs.bpc[s]
    rec["flores_gap_Egyptian"] = bs.bpc["flores:arz_Arab"] / bs.bpc["flores:arb_Arab"]
    rec["flores_gap_Saudi"] = bs.bpc["flores:ars_Arab"] / bs.bpc["flores:arb_Arab"]
    rec["arabic_tax"] = bs.bpc["flores:arb_Arab"] / bs.bpc["flores:eng_Latn"]
    rec["roundtrip"], rec["unk"] = meta["roundtrip_rate"], meta["unk_tokens"]
    rows.append(rec)
S = pd.DataFrame(rows)

# paired bootstrap per dialect (the same document resample for every model)
key = docs[S.model[0]][["set", "doc"]]
for n in S.model:
    assert (docs[n][["set", "doc"]].values == key.values).all(), f"{n}: documents differ"
rng = np.random.default_rng(0)
idx = {s: np.where(key.set.values == s)[0] for s in key.set.unique()}
draws = {s: rng.integers(0, len(ix), size=(1000, len(ix))) for s, ix in idx.items()}


def boot(d, dialect):
    bits, chars = d.bits.values, d.chars.values
    return np.mean([bits[idx[s]][draws[s]].sum(1) / chars[idx[s]][draws[s]].sum(1) for s in sets_of[dialect]], 0)


for d in ARABIC:
    bt = {n: boot(docs[n], d) for n in S.model}
    best = S.model[S[f"bpc_{d}"].idxmin()]
    S[f"rank_{d}"] = S[f"bpc_{d}"].rank().astype(int)
    S[f"lo_{d}"] = [np.percentile(bt[n] - bt[best], 2.5) for n in S.model]
    S[f"hi_{d}"] = [np.percentile(bt[n] - bt[best], 97.5) for n in S.model]
    # models statistically tied with the best (CI of the difference includes 0)
    S[f"tied_{d}"] = S[f"lo_{d}"] <= 0
S.round(4).to_csv(f"{B}/results/summary.csv", index=False)


def size(r):
    return f"{r.params_b:.1f}" + (f" ({r.active_b:g} act.)" if pd.notna(r.active_b) else "")


md = ["# Ranking per dialect", "",
      "BPC = bits per character on held-out text (lower is better; tokenizer-independent). "
      "Δ = difference to the best model with a paired-bootstrap 95% CI; **bold** = not significantly worse than the best. "
      "`post` = only an instruction-tuned checkpoint is public.", ""]
md += ["## Overview (rank by BPC in each variety)", "",
       "| Model | Ckpt | Params (B) | MSA | Egyptian | Saudi | Gulf | Mean dialect rank |", "|---|---|---|---|---|---|---|---|"]
S["mean_dialect_rank"] = S[[f"rank_{d}" for d in DIALECTS]].mean(1)
for _, r in S.sort_values("mean_dialect_rank").iterrows():
    cells = [f"{r[f'bpc_{d}']:.3f} (#{r[f'rank_{d}']})" for d in ARABIC]
    md.append(f"| {r.model} | {r.stage} | {size(r)} | " + " | ".join(cells) + f" | {r.mean_dialect_rank:.1f} |")
for d in DIALECTS + ["MSA"]:
    extra = "FLORES gap" if d in ("Egyptian", "Saudi") else None
    md += ["", f"## {d}", "", f"Sets: {', '.join('`' + s + '`' for s in sets_of[d])}. Benchmark: {MCQ_NAME[d]}.", "",
           f"| # | Model | Ckpt | Params (B) | BPC | Δ vs best (95% CI) | {extra + ' | ' if extra else ''}Fertility (tokens/word) | {MCQ_NAME[d]} acc. |",
           "|---|---|---|---|---|---|" + ("---|" if extra else "") + "---|---|"]
    for i, (_, r) in enumerate(S.sort_values(f"bpc_{d}").iterrows()):
        name = f"**{r.model}**" if r[f"tied_{d}"] else r.model
        delta = "–" if i == 0 else f"+{r[f'bpc_{d}'] - S[f'bpc_{d}'].min():.3f} ({r[f'lo_{d}']:+.3f}, {r[f'hi_{d}']:+.3f})"
        gap = f"{r[f'flores_gap_{d}']:.3f} | " if extra else ""
        md.append(f"| {i + 1} | {name} | {r.stage} | {size(r)} | {r[f'bpc_{d}']:.3f} | {delta} | {gap}{r[f'fert_{d}']:.2f} | {r[f'mcq_{d}']:.1f} |")
md += ["", "Chance accuracy is 25% on both benchmarks. 95% CI half-width: about ±3 points (Belebele, 900 questions) and ±2.8 (Alyah, 1,173)."]
open(f"{B}/results/ranking.md", "w").write("\n".join(md) + "\n")
set_cols = sorted(c for c in S.columns if c.startswith("set:"))
t = S[["model"] + set_cols].rename(columns=lambda c: c.replace("set:", "")).round(3)
open(f"{B}/results/bpc_by_set.md", "w").write(t.to_markdown(index=False) + "\n")
print("\n".join(md))

# figure: one panel per dialect, BPC vs size
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"global": "#2a78d6", "arabic": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharex=True)
fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.13, wspace=0.18)
for ax, d in zip(axes, DIALECTS):
    for _, r in S.iterrows():
        x = r.active_b if pd.notna(r.active_b) else r.params_b
        ax.scatter(x, r[f"bpc_{d}"], s=55, color=COL[r.group], marker="o" if r.stage == "base" else "D", edgecolor=SURFACE, lw=1.5, zorder=3)
        if r[f"rank_{d}"] <= 5 or r.model in ("Llama-3.2-1B",):
            ax.annotate(r.model, (x, r[f"bpc_{d}"]), xytext=(5, 2), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xscale("log"); ax.set_title(d, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Parameters, billions (active for MoE; log)")
axes[0].set_ylabel("Bits per character (lower is better)")
h = [plt.Line2D([], [], ls="", marker="o", color=COL["global"], ms=8), plt.Line2D([], [], ls="", marker="o", color=COL["arabic"], ms=8),
     plt.Line2D([], [], ls="", marker="o", color=INK2, ms=8), plt.Line2D([], [], ls="", marker="D", color=INK2, ms=7)]
fig.legend(h, ["Global model", "Arabic-centric model", "Base checkpoint", "Only post-trained checkpoint public"],
           loc="upper right", bbox_to_anchor=(0.99, 0.93), ncol=4, frameon=False, fontsize=8.5)
fig.suptitle("How well does each model already predict Egyptian, Saudi and Gulf Arabic?", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.015, "Same held-out text for every model. Labels: top 5 per dialect. BPC is tokenizer-independent.", fontsize=8, color=INK2)
fig.savefig(f"{B}/results/bpc_vs_size.png", dpi=200, facecolor=SURFACE)
