"""fig_hero: same score, different script.

  (a) one real prompt, answered by all five models; every clinician-coded
      behaviour drawn as a dot at its position in the reply
  (b) the pooled scores: five labs converge on SCRIPT ~ 0.10, a 0.012-wide
      band, far above the content-driven zero
  (c) but the five position profiles - the script written down - are five
      different signatures; Qwen and Gemini are near-identical twins

Reads out/derived/figdata/hero_data.json (built by 13_figure_data.py).
"""
import json

import numpy as np
import fig_style as S

S.setup()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from paths import FIG, REF

H = json.load(open(REF / "figdata" / "hero_data.json"))
MODELS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
FAM = ["empathy", "advice", "questions"]


def smooth(v):
    x = np.linspace(0.05, 0.95, 10)
    xf = np.linspace(0, 1, 200)
    y = np.interp(xf, x, np.asarray(v, float))
    k = np.exp(-0.5 * np.linspace(-2, 2, 25) ** 2)
    k /= k.sum()
    return xf, np.convolve(np.pad(y, 12, mode="edge"), k, mode="same")[12:-12]


fig = plt.figure(figsize=(12.6, 8.0))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.92], hspace=0.42,
                       left=0.06, right=0.975, top=0.94, bottom=0.075)
g1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                      width_ratios=[1.45, 1.0], wspace=0.16)

# ---- (a) one prompt, five replies --------------------------------------------
axL = fig.add_subplot(g1[0])
S.tag(axL, "a", x=-0.055, y=1.06)
axL.text(0.0, 1.075, "User:  \u201c" + H["user"][:105] + "\u2026\u201d",
         transform=axL.transAxes, fontsize=8.8, style="italic", color=S.MUTE,
         va="bottom")
OFF = {"empathy": 0.18, "advice": 0.0, "questions": -0.18}
for yi, m in enumerate(MODELS):
    y = 4 - yi
    axL.axhline(y, color=S.FAINT, lw=9, zorder=0)
    axL.text(-0.025, y, m, ha="right", va="center", fontsize=10.5,
             fontweight="bold", color=S.MCOL[m])
    for x, g in H["sample"]["events"][m]:
        axL.scatter([x], [y + OFF[g]], s=36, color=S.GCOL[g], alpha=0.95,
                    edgecolor="white", linewidth=0.5, zorder=3)
axL.set_xlim(-0.02, 1.02)
axL.set_ylim(-0.75, 4.62)
axL.set_yticks([])
axL.set_xticks([0, 0.5, 1.0])
axL.set_xticklabels(["start", "mid-reply", "end"], fontsize=9)
S.despine(axL, keep=())
xx = 0.24
for g in FAM:
    axL.scatter([xx], [-0.55], s=36, color=S.GCOL[g], clip_on=False)
    axL.text(xx + 0.018, -0.55, S.GLAB[g], fontsize=9, va="center",
             color=S.INK)
    xx += 0.035 + 0.0155 * len(S.GLAB[g])

# ---- (b) the scores tie -------------------------------------------------------
axS = fig.add_subplot(g1[1])
S.tag(axS, "b", x=-0.03, y=1.06)
vals = {m: H["scores"][m]["SCRIPT"] for m in MODELS}
order = sorted(MODELS, key=lambda m: vals[m])
lo, hi = min(vals.values()), max(vals.values())
axS.axvspan(lo, hi, color="#f6e8c8", zorder=0)
axS.axvline(0, color="#9aa2a9", ls="--", lw=1.2)
axS.text(0, 5.45, "chance", ha="center", va="bottom", fontsize=8.4,
         color=S.MUTE)
for yi, m in enumerate(order):
    axS.plot([0, vals[m]], [yi, yi], color=S.FAINT, lw=2, zorder=1)
    axS.scatter([vals[m]], [yi], s=110, color=S.MCOL[m], zorder=4,
                edgecolor="white", linewidth=1.0)
    axS.text(vals[m] + 0.007, yi, f"{m}  {vals[m]:.3f}", va="center",
             fontsize=9.5, color=S.MCOL[m], fontweight="bold")
axS.annotate("", xy=(hi, 4.85), xytext=(lo, 4.85),
             arrowprops=dict(arrowstyle="<|-|>", color="#b8860b", lw=1.4))
axS.text((lo + hi) / 2, 5.15, f"five labs within {hi - lo:.3f}",
         ha="center", fontsize=9.3, color="#8a6d00", fontweight="bold")
axS.set_xlim(-0.018, 0.155)
axS.set_ylim(-0.8, 5.9)
axS.set_yticks([])
axS.set_xticks([0, 0.05, 0.10, 0.15])
S.despine(axS)
axS.set_xlabel("SCRIPT   (0 = calibrated chance \u00b7 a perfect fixed "
               "template = 0.85)", fontsize=9)

# ---- (c) five fingerprints ----------------------------------------------------
g2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=0.14)
order2 = ["Qwen", "Gemini", "GPT", "Claude", "Llama"]
notes = {"Qwen": "", "Gemini": "", "GPT": "advice-heaviest",
         "Claude": "questions latest,\nand most of them",
         "Llama": "farthest from all"}
ymax = 1.15 * max(max(smooth(H["sig"][m][g])[1].max() for g in FAM)
                  for m in MODELS)
axes2 = []
for j, m in enumerate(order2):
    ax = fig.add_subplot(g2[j])
    axes2.append(ax)
    if j == 0:
        S.tag(ax, "c", x=-0.16, y=1.10)
    for g in FAM:
        xf, y = smooth(H["sig"][m][g])
        ax.fill_between(xf, y, color=S.GCOL[g], alpha=0.42, lw=0)
        ax.plot(xf, y, color=S.GCOL[g], lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["start", "end"], fontsize=8.3)
    S.despine(ax)
    ax.set_title(m, fontsize=11.5, fontweight="bold", color=S.MCOL[m], pad=14)
    if notes[m]:
        ax.text(0.04, 0.97, notes[m], transform=ax.transAxes, ha="left",
                va="top", fontsize=8.2, color=S.MUTE, style="italic")
if True:  # the twin bracket over Qwen + Gemini
    b0 = axes2[0].get_position()
    b1 = axes2[1].get_position()
    yb = b0.y1 + 0.052
    fig.add_artist(Line2D([b0.x0 + 0.012, b1.x1 - 0.012], [yb, yb],
                          color="#8a6d00", lw=1.2))
    for xe in (b0.x0 + 0.012, b1.x1 - 0.012):
        fig.add_artist(Line2D([xe, xe], [yb - 0.012, yb], color="#8a6d00",
                              lw=1.2))
    fig.text((b0.x0 + b1.x1) / 2, yb + 0.008,
             "twins \u2014 JS distance 0.016", ha="center", fontsize=9,
             color="#8a6d00", fontweight="bold")

fig.savefig(FIG / "fig_hero.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_hero.pdf", bbox_inches="tight")
print("saved", FIG / "fig_hero.png")
