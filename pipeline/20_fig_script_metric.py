import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIG, MODELS, OUT, REF, TAB

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}

fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.0], wspace=0.52)

# --- panel A: C + M stacked per model (02_validate_metric.py V1)
val = pd.read_csv(REF / "validation_results.csv").set_index("system")
ax = fig.add_subplot(gs[0])
ms = MODELS
c = [val.loc[m, "C_excess"] for m in ms]
mm = [val.loc[m, "M_excess"] for m in ms]
ax.bar(ms, c, color="#00798c", label="C  (choreography: fixed slots)")
ax.bar(ms, mm, bottom=c, color="#E5A11F", label="M  (momentum: chained moves)")
for i, m in enumerate(ms):
    ax.text(i, c[i] + mm[i] + 0.003, f"z={val.loc[m, 'z']:.0f}", ha="center", fontsize=8.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("SCRIPT  (excess over shuffle null)")
ax.set_ylim(0, 0.135)
n_events = int(val.n_events.sum())
ax.set_title(f"A — Every model runs a template\n(clinician spans, 20 codes, "
             f"{n_events:,} events)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.tick_params(axis="x", labelsize=9)

# --- panel B: the score across annotation schemes. Every anchor is read from
# tables/anchor_*.csv, i.e. recomputed by 11_external_anchors.py on the public
# corpora with this same estimator. Nothing on this panel is a typed-in number.
ax = fig.add_subplot(gs[1])
mt = pd.read_csv(TAB / "anchor_wmt24.csv")
d2t = pd.read_csv(TAB / "anchor_d2t.csv")
rag = pd.read_csv(TAB / "anchor_ragtruth.csv")
control = json.load(open(OUT / "verify_report.json"))["V3"]["SCRIPT"]
coun = [val.loc[m, "SCRIPT"] for m in ms]
yv, labels = [], []


def row(y, vals, color, lab):
    ax.scatter(vals, [y] * len(vals), s=38, color=color, zorder=3,
               edgecolor="k", linewidth=0.4)
    labels.append(lab)
    yv.append(y)


row(4, [control], "#888888", "random control")
row(3, mt.SCRIPT.tolist(), "#aaaaaa", "MT spans\n(WMT24)")
row(2, rag.SCRIPT.tolist(), "#7c9fb0", "hallucination\nspans (RAGTruth)")
row(1, d2t.SCRIPT.tolist(), "#66a182", "d2t error\nspans")
row(0, coun, "#d1495b", "counseling\ncodes")
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(yv)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("SCRIPT")
ax.set_title("B — The instrument has range:\nit stays silent where structure is absent",
             fontsize=10)
ax.annotate(f"z = {val.z.min():.0f}–{val.z.max():.0f}", xy=(0.115, 0.15),
            fontsize=8.5, color="#d1495b")
ax.annotate(f"|z| ≤ {mt.z.abs().max():.1f}", xy=(0.05, 3.2), fontsize=8.5,
            color="#777777")
ax.annotate(f"z ≤ {rag.z.max():.1f}", xy=(0.058, 2.2), fontsize=8.5, color="#7c9fb0")
ax.annotate(f"z ≤ {d2t.z.max():.1f}", xy=(0.062, 1.2), fontsize=8.5, color="#66a182")
ax.set_xlim(-0.05, 0.23)
ax.invert_yaxis()

# --- panel C: per-corpus stability (02_validate_metric.py V2)
sbc = pd.read_csv(REF / "script_by_corpus.csv")
ax = fig.add_subplot(gs[2])
order = ["counselchat", "pair", "carebench", "hope", "ALL"]
for m in ms:
    d = sbc[sbc.model == m].set_index("corpus").loc[order]
    ax.plot(range(5), d.SCRIPT, "-o", ms=4, lw=1.4, color=MCOLORS[m], label=m)
ax.set_xticks(range(5))
ax.set_xticklabels(["counsel-\nchat", "pair", "care-\nbench", "hope", "POOLED"], fontsize=8)
ax.set_ylabel("SCRIPT")
ax.set_title("C — The template travels with the model\n(Claude most scripted in every corpus)", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, ncol=2)
fig.savefig(FIG / "fig_script_metric.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_script_metric.pdf", bbox_inches="tight")
plt.close(fig)
print("saved fig_script_metric")
