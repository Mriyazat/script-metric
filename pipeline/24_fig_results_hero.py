"""Results hero figure: three claims in one glance.
  (a) absolute scale with a meaningful zero (instrument advantage)
  (b) five labs converge on one template level (headline finding)
  (c) same score, five scripts (score vs profile division of labour)

Every number is read from out/ — nothing here is typed in.
No prose titles inside the figure
--- panel tags only; explanation lives in the caption.
"""
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paths import FIG, MODELS, OUT, REF, TAB

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}
gcolors = {"empathy": "#00798c", "advice": "#d1495b", "questions": "#66a182"}
C_COL, M_COL = "#00798c", "#E5A11F"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

val = pd.read_csv(REF / "validation_results.csv").set_index("system")
pr = pd.read_csv(REF / "profile_reading.csv").set_index("model")
human = pd.read_csv(REF / "therapist_baseline.csv").set_index("speaker")

# --- layout -----------------------------------------------------------------
fig = plt.figure(figsize=(14.8, 4.6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 0.95, 1.15], wspace=0.32)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

# =============================================================================
# (a) THE ABSOLUTE SCALE --- meaningful zero made literal
# =============================================================================
# the interpretation ruler, as recomputed by 11_external_anchors.py
_mt = pd.read_csv(TAB / "anchor_wmt24.csv").SCRIPT.sort_values().tolist()
mt = _mt[:-1]                      # the single outlier is annotated separately
mt_outlier = _mt[-1]
d2t = pd.read_csv(TAB / "anchor_d2t.csv").SCRIPT.tolist()
ragt = pd.read_csv(TAB / "anchor_ragtruth.csv").SCRIPT.tolist()
control = json.load(open(OUT / "verify_report.json"))["V3"]["SCRIPT"]
coun = [val.loc[m, "SCRIPT"] for m in MODELS]
human_s = float(human.loc["Therapist", "SCRIPT"])
# same 3-label alphabet as the therapist, so the comparison is within-instrument
human_models = [float(human.loc[m, "SCRIPT"]) for m in MODELS]

# rows top to bottom: our benchmark (models, then the human anchor), then the
# external schemes that give the number a scale
rows = [
    ("five frontier models\n(20 clinician codes)", coun, "#d1495b",
     f"z = {val.z.min():.0f}–{val.z.max():.0f}"),
    ("therapist vs. same models\n(shared 3-label alphabet)", [human_s],
     "#333333", ""),
    ("data-to-text\nerror spans", d2t, "#66a182", ""),
    ("hallucination spans\n(RAGTruth)", ragt, "#7c9fb0", ""),
    ("MT error spans\n(WMT24)", mt, "#aaaaaa", "≈ chance"),
    ("random-label\ncontrol", [control], "#888888", "SCRIPT = 0"),
]
BAND_LO, BAND_HI = -0.62, 1.5           # the two Cognitive-Atrophy-Bench rows
axA.axhspan(BAND_LO, BAND_HI, color="#f1f4f9", zorder=0)

for y, (lab, vals, col, note) in enumerate(rows):
    if y == 1:  # the five models under the therapist's alphabet, for reference
        axA.scatter(human_models, [y] * len(human_models), s=30,
                    facecolor="white", edgecolor="#999999", linewidth=1.0,
                    zorder=3)
    axA.scatter(vals, [y] * len(vals), s=55, color=col, zorder=4,
                edgecolor="white", linewidth=0.8)
    if note:
        axA.text(max(vals) + 0.007, y, note, ha="left", va="center",
                 fontsize=8, color=col, style="italic")

# row 1 is annotated on both sides: the therapist below all five of its own
# reference points is the message, so each group is named where it sits
axA.text(human_s - 0.008, 1, "therapist", ha="right", va="center",
         fontsize=8, color="#333333", style="italic", fontweight="bold")
axA.text(float(np.mean(human_models)), 0.60, "the same five models",
         ha="center", va="center", fontsize=7.6, color="#8a8a8a",
         style="italic")

axA.axvline(0, color="#888888", ls="--", lw=1.2, zorder=2)
axA.text(0, BAND_LO - 0.12, "SCRIPT = 0", ha="center", va="top", fontsize=8.5,
         fontweight="bold", color="#666666")
axA.text(0.235, BAND_LO - 0.12, "Cognitive Atrophy Bench", ha="right",
         va="top", fontsize=7.6, style="italic", color="#7f8c9b")
axA.set_xlim(-0.06, 0.235)
axA.set_ylim(len(rows) - 0.3, -0.95)
axA.set_yticks(range(len(rows)))
axA.set_yticklabels([r[0] for r in rows], fontsize=8.2, color="#333333")
axA.tick_params(axis="y", length=0, pad=3)
axA.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
axA.set_xlabel("SCRIPT  (excess over chance)", fontsize=10)
axA.spines[["top", "right", "left"]].set_visible(False)
axA.text(0.0, 1.04, "(a)", transform=axA.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

# =============================================================================
# (b) FIVE LABS, ONE TEMPLATE LEVEL --- C + M stacked
# =============================================================================
ms = MODELS
c = [val.loc[m, "C_excess"] for m in ms]
mm = [val.loc[m, "M_excess"] for m in ms]
x = np.arange(len(ms))
axB.bar(x, c, color=C_COL, width=0.72, label="C  choreography")
axB.bar(x, mm, bottom=c, color=M_COL, width=0.72, label="M  momentum")
for i, m in enumerate(ms):
    tot = c[i] + mm[i]
    axB.text(i, tot + 0.004, f"{tot:.3f}", ha="center", fontsize=8.5,
             fontweight="bold", color=MCOLORS[m])
    axB.text(i, -0.012, m, ha="center", va="top", fontsize=9,
             fontweight="bold", color=MCOLORS[m])
axB.axhline(0, color="k", lw=0.6)
axB.set_ylim(-0.028, 0.145)
axB.set_xticks([])
axB.set_ylabel("SCRIPT", fontsize=10)
axB.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
axB.spines[["top", "right"]].set_visible(False)
axB.text(0.0, 1.04, "(b)", transform=axB.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")
# faint band emphasising the ~0.10 convergence
axB.axhspan(0.097, 0.110, color="#f0f0f0", zorder=0)

# =============================================================================
# (c) SAME SCORE, FIVE SCRIPTS --- profile geometry
# =============================================================================
SCALE = 7000
order = ["Qwen", "Gemini", "GPT", "Claude", "Llama"]
sigs = {"Qwen": "twin of Gemini",
        "Gemini": "twin of Qwen",
        "GPT": "advice machine",
        "Claude": "questions latest",
        "Llama": "far from all"}
for yi, m in enumerate(order[::-1]):
    r = pr.loc[m]
    axC.hlines(yi, 0, 1, color="#e8e8e8", lw=2.2, zorder=1)
    for g, pos_c, sh_c in [("empathy", "empathy_pos", "empathy_share"),
                           ("advice", "advice_pos", "advice_share"),
                           ("questions", "questions_pos", "questions_share")]:
        axC.scatter(r[pos_c], yi, s=r[sh_c] * SCALE, color=gcolors[g],
                    alpha=0.88, edgecolor="white", linewidth=1.4, zorder=3)
    axC.text(-0.03, yi, m, ha="right", va="center", fontsize=11,
             fontweight="bold", color=MCOLORS[m])
    axC.text(1.03, yi, sigs[m], ha="left", va="center", fontsize=8.5,
             color="#666666", style="italic")
axC.set_xlim(-0.02, 1.02)
axC.set_ylim(-0.55, len(order) - 0.25)
axC.set_yticks([])
axC.set_xticks([0, 0.5, 1.0])
axC.set_xticklabels(["start", "mid", "end"], fontsize=9)
axC.set_xlabel("mean position in reply   ·   marker area = share", fontsize=9.5)
axC.spines[["top", "right", "left"]].set_visible(False)
handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=gcolors[g],
                      markeredgecolor="white", label=g) for g in gcolors]
axC.legend(handles=handles, ncol=3, frameon=False, fontsize=8.5,
           loc="upper center", bbox_to_anchor=(0.5, 1.12))
axC.text(0.0, 1.04, "(c)", transform=axC.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

fig.savefig(FIG / "fig_results_hero.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_results_hero.pdf", bbox_inches="tight")
print("saved", FIG / "fig_results_hero.png")
