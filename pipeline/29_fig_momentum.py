"""fig_momentum: what the momentum term measures, through one code pair.

  (a) the habit, verbatim: three real VIN -> DIR hand-offs (validate, then
      advise) from three different Claude replies
  (b) the measurement: the distribution of what comes after a VIN span
      (filled dots) against what position alone would predict at those same
      slots (open dots); the DIR surplus is momentum that choreography
      cannot explain
  (c) the sum: the strongest pulls across all code pairs; VIN -> DIR is one
      of many, and together they make up M

Reads out/derived/figdata/mom.json (built by 13_figure_data.py).
"""
import json

import numpy as np
import fig_style as S

S.setup()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paths import FIG, REF

M = json.load(open(REF / "figdata" / "mom.json"))
CODES, FAM = M["codes"], M["fam"]
obs, base = np.array(M["obs"]), np.array(M["base"])
ci = {c: k for k, c in enumerate(CODES)}
NAVY, GREY = "#2e4057", "#8d96a3"          # ltc minou accents

fig = plt.figure(figsize=(11.6, 10.6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.35, 2.3, 2.0], hspace=0.52,
                       left=0.10, right=0.965, top=0.965, bottom=0.06)

# ---- (a) the habit, verbatim ---------------------------------------------------
axA = fig.add_subplot(gs[0])
axA.axis("off")
S.tag(axA, "a", x=-0.075, y=0.98)
for r, (a, b) in enumerate(M["examples"][:3]):
    y = 0.82 - r * 0.36
    axA.add_patch(FancyBboxPatch((0.0, y - 0.13), 0.435, 0.26,
                  boxstyle="round,pad=0.006,rounding_size=0.02",
                  fc=S.GCOL["empathy"], ec="none", alpha=0.16,
                  transform=axA.transAxes))
    axA.text(0.012, y, "VIN \u00b7 \u201c" + a.strip() + "\u2026\u201d",
             transform=axA.transAxes, fontsize=8.6, va="center",
             color="#00565f", style="italic")
    axA.add_patch(FancyArrowPatch((0.45, y), (0.525, y),
                  transform=axA.transAxes, arrowstyle="-|>",
                  mutation_scale=15, color=S.INK, lw=1.5))
    axA.add_patch(FancyBboxPatch((0.54, y - 0.13), 0.45, 0.26,
                  boxstyle="round,pad=0.006,rounding_size=0.02",
                  fc=S.GCOL["advice"], ec="none", alpha=0.16,
                  transform=axA.transAxes))
    axA.text(0.552, y, "DIR \u00b7 \u201c" + b.strip() + "\u2026\u201d",
             transform=axA.transAxes, fontsize=8.6, va="center",
             color="#96323c", style="italic")
axA.text(0.0, -0.14, "VIN = validation (empathy)", transform=axA.transAxes,
         fontsize=8.8, color=S.GCOL["empathy"], fontweight="bold")
axA.text(0.54, -0.14, "DIR = direct advice", transform=axA.transAxes,
         fontsize=8.8, color=S.GCOL["advice"], fontweight="bold")

# ---- (b) what follows VIN, vs what position predicts ---------------------------
axB = fig.add_subplot(gs[1])
S.tag(axB, "b", x=-0.075, y=1.02)
show = ["DIR", "VIN", "QCL", "FIX", "TSH", "TEN", "AUR"]
order = sorted(show, key=lambda c: obs[ci[c]])
for k, c in enumerate(order):
    bs, ob = base[ci[c]], obs[ci[c]]
    axB.plot([bs, ob], [k, k], color="#c3c9ce", lw=2.2, zorder=1)
    axB.scatter([bs], [k], s=75, facecolor="white", edgecolor=GREY,
                linewidth=1.4, zorder=3)
    axB.scatter([ob], [k], s=100, color=S.GCOL[FAM[c]], edgecolor="white",
                linewidth=0.8, zorder=4)
    axB.text(-0.012, k, c, ha="right", va="center", fontsize=9.5,
             color=S.GCOL[FAM[c]], fontweight="bold")
    if c == "DIR":
        axB.annotate("", xy=(ob, k - 0.38), xytext=(bs, k - 0.38),
                     arrowprops=dict(arrowstyle="<|-|>", color="#96323c",
                                     lw=1.7))
        axB.text((bs + ob) / 2, k - 0.52,
                 f"+{(ob - bs) * 100:.0f} points of advice that position "
                 "did not predict",
                 ha="center", va="top", fontsize=9.5, color="#96323c",
                 fontweight="bold")
axB.set_yticks([])
axB.set_xlim(0.0, max(obs[[ci[c] for c in show]]) * 1.14)
axB.set_ylim(-1.5, len(order) - 0.4)
S.despine(axB)
axB.set_xlabel("probability of being the next code after a VIN span "
               f"($n$ = {M['nvin']})", fontsize=9.3)
axB.scatter([], [], s=75, facecolor="white", edgecolor=GREY, linewidth=1.4,
            label="what position alone predicts at those slots")
axB.scatter([], [], s=100, color=S.INK,
            label="what actually follows VIN  (colour = family)")
axB.legend(loc="lower right", fontsize=8.8, frameon=False)

# ---- (c) every pull, summed = M ------------------------------------------------
axC = fig.add_subplot(gs[2])
S.tag(axC, "c", x=-0.075, y=1.02)
pulls = M["pulls"]
names = [f"{a} \u2192 {b}" for a, b, s_, n_ in pulls][::-1]
surp = [s_ for a, b, s_, n_ in pulls][::-1]
kind = [("focus" if (a == "VIN" and b == "DIR") else
         "cross" if a != b else "self") for a, b, s_, n_ in pulls][::-1]
KCOL = {"focus": S.GCOL["advice"], "cross": NAVY, "self": "#c7ccd1"}
for k in range(len(names)):
    axC.barh(k, surp[k], color=KCOL[kind[k]], height=0.66, zorder=3)
    axC.text(surp[k] + 0.008, k, f"+{surp[k]:.2f}", va="center", fontsize=8.4,
             color=S.MUTE)
axC.set_yticks(range(len(names)))
axC.set_yticklabels(names, fontsize=9)
for t, kd in zip(axC.get_yticklabels(), kind):
    t.set_color(KCOL[kd] if kd != "self" else "#6a737b")
    if kd == "focus":
        t.set_fontweight("bold")
axC.set_xlim(0, max(surp) * 1.16)
S.despine(axC, keep=("bottom", "left"))
axC.set_xlabel("surplus predictability of the next code, beyond position",
               fontsize=9.3)
axC.text(0.985, 0.06,
         "the VIN \u2192 DIR pull from (b)\n"
         "other cross-code pulls\n"
         "a code repeating itself",
         transform=axC.transAxes, fontsize=8.8, va="bottom", ha="right",
         color=S.INK, linespacing=1.7)
for dyk, kd in zip((0.208, 0.135, 0.062), ("focus", "cross", "self")):
    axC.add_patch(plt.Rectangle((0.680, 0.06 + dyk - 0.020), 0.022, 0.045,
                  transform=axC.transAxes, fc=KCOL[kd], ec="none",
                  clip_on=False))
axC.text(0.985, 0.40, f"summed over every pair:  $M$ = {M['M_excess']:.3f}",
         transform=axC.transAxes, fontsize=9.6, va="bottom", ha="right",
         color=S.INK, fontweight="bold")

fig.savefig(FIG / "fig_momentum.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_momentum.pdf", bbox_inches="tight")
print("saved", FIG / "fig_momentum.png",
      "VIN->DIR %+.3f" % (obs[ci["DIR"]] - base[ci["DIR"]]))
