"""fig_worked_example: one real reply -> the number, as a four-stage pipeline.

  (a) the clinician's highlights on one real Claude reply
  (b) the reduction: every highlight becomes one dot at its start position
      (dashed connectors show the mapping; shaded stripes are the 10 bins)
  (c) pooling all 816 Claude replies into the two count tables the metric
      reads: where each behaviour sits (-> C) and what follows what (-> M)
  (d) calibration: the observed structure R against the within-reply shuffle
      null; the excess is SCRIPT, and it splits exactly into C + M on the
      same axis

Reads out/derived/figdata/{fig_reply,fig_pooled}.json (built by 13_figure_data.py).
"""
import json

import numpy as np
import fig_style as S

S.setup()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from paths import FIG, REF

R = json.load(open(REF / "figdata" / "fig_reply.json"))
P = json.load(open(REF / "figdata" / "fig_pooled.json"))
GG = ["empathy", "advice", "questions", "other"]
# painting priority for overlapping highlights (advice spans are often long
# and would otherwise blanket the empathy spans they contain)
PRIO = {"questions": 4, "empathy": 3, "advice": 2, "other": 1}

txt, L, spans = R["txt"], R["L"], R["spans"]

fig = plt.figure(figsize=(9.8, 12.2))
gs = gridspec.GridSpec(4, 1, height_ratios=[2.15, 0.95, 1.85, 1.35],
                       hspace=0.62, left=0.085, right=0.965,
                       top=0.97, bottom=0.055)

# ================================ (a) the reply ================================
axA = fig.add_subplot(gs[0])
axA.axis("off")
S.tag(axA, "a", x=-0.065, y=0.985)
axA.text(0.0, 1.03, "User:  \u201c" + R["user"][:68]
         + ("\u2026\u201d" if len(R["user"]) > 68 else "\u201d"),
         transform=axA.transAxes, fontsize=8.8, style="italic",
         color=S.MUTE, va="bottom")
x0, y0, dy = 0.006, 0.885, 0.117
line_ranges, cw = S.draw_reply(axA, txt, L, spans, PRIO, wrap=100, x0=x0,
                               y0=y0, dy=dy, fontsize=9.7)

# colour key, on the top row next to the user's message: coloured words,
# matching how the annotation is rendered in the text itself
xx, yk = 0.62, 1.045
for g in GG:
    axA.text(xx, yk, S.GLAB[g], transform=axA.transAxes, fontsize=9,
             va="center", fontweight="bold",
             color="#8d96a3" if g == "other" else S.GCOL[g])
    axA.text(xx - 0.014, yk, "\u25cf", transform=axA.transAxes, fontsize=7,
             va="center", color="#8d96a3" if g == "other" else S.GCOL[g])
    xx += 0.028 + 0.0105 * len(S.GLAB[g])
axA.text(0.62, 0.975, "faint text = not annotated", transform=axA.transAxes,
         fontsize=8, va="center", style="italic", color="#9aa2a9")

# ================================ (b) the events ===============================
axB = fig.add_subplot(gs[1])
S.tag(axB, "b", x=-0.065, y=1.06)
LANE = {"empathy": 3.0, "advice": 2.0, "questions": 1.0, "other": 0.25}
axB.set_xlim(-0.005, 1.005)
axB.set_ylim(-0.65, 3.75)
for b in range(0, 10, 2):                       # the 10 position bins
    axB.axvspan(b / 10, (b + 1) / 10, color="#f2f4f6", zorder=0)
for b in range(11):
    axB.axvline(b / 10, color="#dfe3e7", lw=0.7, zorder=1)
for g, y in LANE.items():
    axB.text(-0.015, y, S.GLAB[g], ha="right", va="center", fontsize=8.8,
             color=S.GCOL[g], fontweight="bold")
    axB.axhline(y, color=S.FAINT, lw=5.5, zorder=1)
for s in spans:
    g = s["group"]
    axB.scatter([s["x"]], [LANE[g]], s=52 if g != "other" else 30,
                color=S.GCOL[g], zorder=4, edgecolor="white", linewidth=0.7)
axB.set_yticks([])
S.despine(axB)
axB.set_xticks([0, 0.5, 1.0])
axB.set_xticklabels(["start", "mid-reply", "end"], fontsize=8.8)
axB.set_xlabel("start position of the highlight (10 shaded bins)", fontsize=9)

# numbered badges: the same three spans marked once where they start in the
# text and once above the dot they become, so the mapping needs no arrows
def badge(ax, x, y, num, color, coords):
    ax.text(x, y, str(num), transform=coords, fontsize=6.6, color="white",
            ha="center", va="center", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.24", fc=color, ec="white",
                      lw=0.6))

cand = {g: [si for si, s in enumerate(spans) if s["group"] == g]
        for g in ("empathy", "advice", "questions")}
picks = [min(cand["empathy"], key=lambda si: spans[si]["x"]),
         min(cand["advice"], key=lambda si: abs(spans[si]["x"] - 0.55)),
         min(cand["questions"], key=lambda si: spans[si]["x"])]
for num, si in enumerate(picks, 1):
    s = spans[si]
    for a, b, yl in line_ranges:
        if a <= s["start"] < b:
            badge(axA, x0 + (s["start"] - a) * cw + 0.004, yl + 0.052, num,
                  S.GCOL[s["group"]], axA.transAxes)
            break
    badge(axB, s["x"], LANE[s["group"]] + 0.48, num, S.GCOL[s["group"]],
          axB.transData)

# ================================ (c) the tables ===============================
g2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                      width_ratios=[1.35, 1.0], wspace=0.32)
axP = fig.add_subplot(g2[0])
axT = fig.add_subplot(g2[1])
S.tag(axP, "c", x=-0.088, y=1.10)

pos = np.array(P["pos"])
posn = pos / pos.sum(1, keepdims=True)
axP.imshow(posn, aspect="auto", cmap=S.ltc_seq, vmin=0, vmax=0.32)
axP.set_xticks(range(10))
axP.set_xticklabels([f".{i}" for i in range(10)], fontsize=7.5)
axP.set_yticks(range(4))
axP.set_yticklabels([S.GLAB[g] for g in GG], fontsize=9)
for t, g in zip(axP.get_yticklabels(), GG):
    t.set_color(S.GCOL[g])
    t.set_fontweight("bold")
axP.set_xlabel("position bin", fontsize=8.8)
axP.set_title("where each behaviour sits   $\\rightarrow\\ C$",
              fontsize=10, pad=5, color=S.INK)
for (r_, c_), v in np.ndenumerate(posn):
    if v >= 0.12:
        axP.text(c_, r_, f"{v:.0%}", ha="center", va="center", fontsize=6.8,
                 color="white" if v > 0.20 else S.INK)

tr = np.array(P["tr"])
trn = tr / tr.sum(1, keepdims=True)
axT.imshow(trn, aspect="auto", cmap=S.ltc_seq, vmin=0, vmax=0.55)
axT.set_xticks(range(4))
axT.set_xticklabels([S.GLAB[g] for g in GG], fontsize=8, rotation=30,
                    ha="right")
axT.set_yticks(range(4))
axT.set_yticklabels([S.GLAB[g] for g in GG], fontsize=9)
for t, g in zip(axT.get_yticklabels(), GG):
    t.set_color(S.GCOL[g])
    t.set_fontweight("bold")
for t, g in zip(axT.get_xticklabels(), GG):
    t.set_color(S.GCOL[g])
axT.set_xlabel("next behaviour", fontsize=8.8)
axT.set_title("what follows what   $\\rightarrow\\ M$",
              fontsize=10, pad=5, color=S.INK)
for (r_, c_), v in np.ndenumerate(trn):
    axT.text(c_, r_, f"{v:.0%}", ha="center", va="center", fontsize=7.6,
             color="white" if v > 0.33 else S.INK,
             fontweight="bold" if r_ == c_ else "normal")

# ================================ (d) calibration ==============================
axD = fig.add_subplot(gs[3])
S.tag(axD, "d", x=-0.065, y=1.02)
sc = P["score"]
Robs, Rnull = sc["R_raw"], sc["R_null"]
script, z = sc["SCRIPT"], sc["z"]
C, M = sc["C_excess"], sc["M_excess"]
sd = script / z
xs = np.linspace(Rnull - 7 * sd, Robs + 9 * sd, 600)
null = np.exp(-0.5 * ((xs - Rnull) / sd) ** 2)
axD.fill_between(xs, null, color="#d9dde1", zorder=2)
axD.plot(xs, null, color="#b3b9bf", lw=0.9, zorder=2)
axD.text(Rnull, 1.06, "chance\n(within-reply shuffle)", ha="center",
         fontsize=8.2, color=S.MUTE, va="bottom")
# the observed marker stops above the C+M bar, whose right edge is the
# observed value itself
axD.plot([Robs, Robs], [0.42, 1.24], color=S.INK, lw=2.2, zorder=4,
         solid_capstyle="butt")
axD.text(Robs, 1.28, "observed", ha="center", fontsize=8.8, color=S.INK,
         fontweight="bold", va="bottom")

# the excess itself, decomposed on the same axis: SCRIPT = C + M
y0b, hb = 0.16, 0.20
axD.add_patch(plt.Rectangle((Rnull, y0b), C, hb, fc=S.GCOL["empathy"],
                            ec="white", lw=0.8, zorder=4))
axD.add_patch(plt.Rectangle((Rnull + C, y0b), M, hb, fc=S.GCOL["advice"],
                            ec="white", lw=0.8, zorder=4))
axD.text(Rnull + C / 2, y0b - 0.09, f"$C$ = {C:.3f}\nfixed seats",
         ha="center", va="top", fontsize=8.6, color=S.GCOL["empathy"],
         fontweight="bold", linespacing=1.15)
axD.text(Rnull + C + M / 2, y0b - 0.09, f"$M$ = {M:.3f}\nchained moves",
         ha="center", va="top", fontsize=8.6, color=S.GCOL["advice"],
         fontweight="bold", linespacing=1.15)
axD.annotate("", xy=(Robs, 0.62), xytext=(Rnull, 0.62),
             arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=1.5))
axD.text((Robs + Rnull) / 2, 0.68,
         f"SCRIPT = {script:.3f}    ($z$ = {z:.0f})",
         ha="center", fontsize=11.5, fontweight="bold", color=S.INK)
axD.text(Robs + 2.5 * sd, 0.30,
         f"{int(sc['n_events']):,} events\n{int(sc['n_responses'])} replies",
         fontsize=8.4, color=S.MUTE, va="center")
axD.set_xlim(xs[0], xs[-1])
axD.set_ylim(-0.55, 1.55)
axD.set_yticks([])
S.despine(axD)
axD.set_xlabel("$R$   (share of behaviour uncertainty resolved by structure alone)",
               fontsize=9)

# ---- flow verbs in the gaps between stages -----------------------------------
def between(ax_top, ax_bot):
    return (ax_top.get_position().y0 + ax_bot.get_position().y1) / 2

S.flow(fig, between(axA, axB) + 0.002,
       "keep one event per highlight:  (behaviour, start position)", x=0.30)
S.flow(fig, between(axB, axP) - 0.006,
       "pool the events of all 816 Claude replies", x=0.30)
S.flow(fig, between(axP, axD) - 0.008,
       "shuffle labels within each reply \u2192 chance", x=0.30)

fig.savefig(FIG / "fig_worked_example.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_worked_example.pdf", bbox_inches="tight")
print("saved", FIG / "fig_worked_example.png")
