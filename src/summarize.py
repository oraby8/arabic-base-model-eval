"""Summarise every finished model: results/summary.csv, results/ranking.md, results/bpc_vs_size.png.

BPC "MSA" / "dialects" = mean over the three sources (FLORES+, PADIC, FineWeb-2) of that source's MSA set /
unweighted mean of its dialect sets, so no source dominates because it has more dialect sets.
dialect gap = dialect BPC / MSA BPC.  Arabic tax = FLORES MSA BPC / FLORES English BPC on the same (parallel) content.
CI: paired bootstrap over documents (1,000 resamples, the same resample for every model), for the
difference in dialect BPC between each model and the best one.
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
from models import MODELS, is_msa

R = f"{B}/results/models"
done = [m for m in MODELS if os.path.exists(f"{R}/{m['name']}/meta.json")]
SRC = ["flores", "padic", "fineweb"]


def split_sets(sets):
    out = {}
    for s in sets:
        src, v = s.split(":")
        if v == "eng_Latn":
            continue
        out.setdefault(src, {"msa": [], "dia": []})["msa" if is_msa(v) else "dia"].append(s)
    return out


rows, docs = [], {}
for m in done:
    n = m["name"]
    meta = json.load(open(f"{R}/{n}/meta.json"))
    b = pd.read_csv(f"{R}/{n}/bpc.csv").set_index("set")
    bel = pd.read_csv(f"{R}/{n}/belebele.csv").set_index("lang")
    docs[n] = pd.read_parquet(f"{R}/{n}/docs.parquet")
    groups = split_sets(b.index)
    rec = dict(model=n, group=m["group"], stage=m["stage"], params_b=meta["n_params"] / 1e9, active_b=m.get("active_b"))
    for src in SRC:
        rec[f"{src}_msa"] = b.bpc[groups[src]["msa"]].mean()
        rec[f"{src}_dia"] = b.bpc[groups[src]["dia"]].mean()
    rec["bpc_msa"] = np.mean([rec[f"{s}_msa"] for s in SRC])
    rec["bpc_dia"] = np.mean([rec[f"{s}_dia"] for s in SRC])
    rec["dialect_gap"] = rec["bpc_dia"] / rec["bpc_msa"]
    rec["bpc_eng"] = b.bpc["flores:eng_Latn"]
    rec["arabic_tax"] = b.bpc["flores:arb_Arab"] / rec["bpc_eng"]
    rec["fert_msa"] = b.fertility["flores:arb_Arab"]
    rec["fert_dia"] = b.fertility[groups["flores"]["dia"]].mean()
    rec["cpt_msa"] = b.chars_per_token["flores:arb_Arab"]
    rec["fert_eng"] = b.fertility["flores:eng_Latn"]
    rec["bel_msa"] = 100 * bel.acc["arb_Arab"]
    rec["bel_dia"] = 100 * bel.acc.drop(["arb_Arab", "eng_Latn"]).mean()
    rec["bel_eng"] = 100 * bel.acc["eng_Latn"]
    rec["roundtrip"] = meta["roundtrip_rate"]
    rec["unk"] = meta["unk_tokens"]
    rows.append(rec)
S = pd.DataFrame(rows).sort_values("bpc_dia").reset_index(drop=True)

# paired bootstrap of dialect BPC (same doc resample for all models)
key = docs[S.model[0]][["set", "doc"]]
sets = key.set.values
rng = np.random.default_rng(0)
idx_by_set = {s: np.where(sets == s)[0] for s in np.unique(sets)}
groups = split_sets(idx_by_set)
REPS = 1000
draws = {s: rng.integers(0, len(ix), size=(REPS, len(ix))) for s, ix in idx_by_set.items()}


def boot_dia(d):
    bits, chars = d.bits.values, d.chars.values
    per_set = {s: bits[ix][draws[s]].sum(1) / chars[ix][draws[s]].sum(1) for s, ix in idx_by_set.items()}
    return np.mean([np.mean([per_set[s] for s in groups[src]["dia"]], 0) for src in SRC], 0)


for n in S.model:
    assert (docs[n][["set", "doc"]].values == key.values).all(), f"{n}: documents differ"
boots = {n: boot_dia(docs[n]) for n in S.model}
best = S.model[0]
S["dia_diff_vs_best_lo"] = [np.percentile(boots[n] - boots[best], 2.5) for n in S.model]
S["dia_diff_vs_best_hi"] = [np.percentile(boots[n] - boots[best], 97.5) for n in S.model]
S.round(4).to_csv(f"{B}/results/summary.csv", index=False)

# markdown ranking
lines = ["| # | Model | Group | Ckpt | Params (B) | BPC MSA | BPC dialects | Δ vs best (95% CI) | Dialect gap | Arabic tax | Fertility MSA / dialects | Belebele MSA / dialects / Eng |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in S.iterrows():
    p = f"{r.params_b:.1f}" + (f" ({r.active_b:g} act.)" if pd.notna(r.active_b) else "")
    d = "–" if r.model == best else f"+{r.bpc_dia - S.bpc_dia[0]:.3f} ({r.dia_diff_vs_best_lo:+.3f}, {r.dia_diff_vs_best_hi:+.3f})"
    lines.append(f"| {i + 1} | {r.model} | {r.group} | {r.stage} | {p} | {r.bpc_msa:.3f} | {r.bpc_dia:.3f} | {d} | {r.dialect_gap:.3f} | "
                 f"{r.arabic_tax:.2f} | {r.fert_msa:.2f} / {r.fert_dia:.2f} | {r.bel_msa:.1f} / {r.bel_dia:.1f} / {r.bel_eng:.1f} |")
open(f"{B}/results/ranking.md", "w").write("\n".join(lines) + "\n")
src_cols = ["model"] + [f"{s}_{k}" for s in SRC for k in ("msa", "dia")]
open(f"{B}/results/bpc_by_source.md", "w").write(S[src_cols].round(3).to_markdown(index=False) + "\n")
print("\n".join(lines))

# figure: dialect BPC vs size
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"global": "#2a78d6", "arabic": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)
for _, r in S.iterrows():
    x = r.active_b if pd.notna(r.active_b) else r.params_b
    ax.scatter(x, r.bpc_dia, s=70, color=COL[r.group], marker="o" if r.stage == "base" else "D",
               edgecolor=SURFACE, lw=2, zorder=3)
    ax.annotate(r.model, (x, r.bpc_dia), xytext=(6, 3), textcoords="offset points", fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xlabel("Parameters, billions (active parameters for MoE; log scale)")
ax.set_ylabel("Bits per character on dialects (lower is better)")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
h = [plt.Line2D([], [], ls="", marker="o", color=COL["global"], ms=8), plt.Line2D([], [], ls="", marker="o", color=COL["arabic"], ms=8),
     plt.Line2D([], [], ls="", marker="o", color=INK2, ms=8), plt.Line2D([], [], ls="", marker="D", color=INK2, ms=7)]
fig.legend(h, ["Global model", "Arabic-centric model", "Base checkpoint", "Only post-trained checkpoint public"],
           loc="upper right", bbox_to_anchor=(0.99, 0.92), ncol=4, frameon=False, fontsize=8.5)
fig.suptitle("How well does each model already predict Arabic dialects?", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.015, "Same held-out text for every model: FLORES+, PADIC, FineWeb-2 test (ALDi-filtered). BPC is tokenizer-independent.",
         fontsize=8, color=INK2)
fig.savefig(f"{B}/results/bpc_vs_size.png", dpi=200, facecolor=SURFACE)
