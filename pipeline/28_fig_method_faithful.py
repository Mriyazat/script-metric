"""fig_method_faithful: the metric at full resolution, nothing hidden.

  (a) the same real reply as the worked example, but showing all 20
      clinician codes (colour = family; the score sees every code)
  (b) the reduction to events, code by code, on the 10 position bins
  (c) the two tables actually consumed by the metric: the 20x10 position
      table (-> C) and the 20x20 transition table (-> M), pooled over all
      816 Claude replies; white rules separate the families
  (d) the score those two tables produce, decomposed C + M

Reads out/derived/figdata/{faithful,fig_reply}.json (built by 13_figure_data.py).
"""
import json

import numpy as np
import fig_style as S

S.setup()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paths import FIG, REF

F = json.load(open(REF / "figdata" / "faithful.json"))
R = json.load(open(REF / "figdata" / "fig_reply.json"))
GG = ["empathy", "advice", "questions", "other"]
PRIO = {"questions": 4, "empathy": 3, "advice": 2, "other": 1}

labels, fam = F["labels"], F["fam"]
pos, tr = np.array(F["pos"]), np.array(F["tr"])
n = len(labels)
bnd = [i for i in range(1, n) if fam[i] != fam[i - 1]]

txt, L, spans = R["txt"], R["L"], R["spans"]

fig = plt.figure(figsize=(12.6, 14.4))
gs = gridspec.GridSpec(4, 1, height_ratios=[2.15, 1.15, 4.6, 0.72],
                       hspace=0.44, left=0.09, right=0.97,
                       top=0.975, bottom=0.035)

# ---- (a) the reply, all 20 codes ---------------------------------------------
axA = fig.add_subplot(gs[0])
axA.axis("off")
S.tag(axA, "a", x=-0.075, y=0.97)
axA.text(0.0, 1.005, "User:  \u201c" + R["user"][:70] + "\u2026\u201d",
         transform=axA.transAxes, fontsize=8.8, style="italic", color=S.MUTE,
         va="bottom")
line_ranges, cw = S.draw_reply(axA, txt, L, spans, PRIO, wrap=132, x0=0.004,
                               y0=0.865, dy=0.135, fontsize=9.6)

# stamp each span's actual code above the character where it starts, so the
# 20-code layer is visible in the reply itself; codes whose spans start at
# (nearly) the same character slide right to sit side by side
last_end = {}
for s in sorted(spans, key=lambda s: s["start"]):
    for li, (a, b, yl) in enumerate(line_ranges):
        if a <= s["start"] < b:
            xa = 0.004 + (s["start"] - a) * cw
            if li in last_end and xa < last_end[li]:
                xa = last_end[li]
            last_end[li] = xa + 0.0044 * len(s["code"]) + 0.006
            axA.text(xa, yl + 0.044, s["code"], transform=axA.transAxes,
                     fontsize=5.9, va="bottom", ha="left", fontweight="bold",
                     color=("#8d96a3" if s["group"] == "other"
                            else S.GCOL[s["group"]]), zorder=3)
            break
xx = 0.62
for g in GG:
    axA.text(xx, 1.005, S.GLAB[g], transform=axA.transAxes, fontsize=9,
             va="bottom", fontweight="bold",
             color="#8d96a3" if g == "other" else S.GCOL[g])
    axA.text(xx - 0.013, 1.012, "\u25cf", transform=axA.transAxes,
             fontsize=6.5, va="bottom",
             color="#8d96a3" if g == "other" else S.GCOL[g])
    xx += 0.026 + 0.0098 * len(S.GLAB[g])
axA.text(0.62, 0.955, "faint text = not annotated", transform=axA.transAxes,
         fontsize=7.8, va="bottom", style="italic", color="#9aa2a9")

# ---- (b) the events, with their codes -----------------------------------------
axB = fig.add_subplot(gs[1])
S.tag(axB, "b", x=-0.075, y=1.04)
axB.set_xlim(-0.005, 1.005)
axB.set_ylim(-1.15, 1.85)
for b in range(0, 10, 2):
    axB.axvspan(b / 10, (b + 1) / 10, color="#f2f4f6", zorder=0)
for b in range(11):
    axB.axvline(b / 10, color="#dfe3e7", lw=0.7, zorder=1)
LANE = {"empathy": 0.55, "advice": 0.0, "questions": -0.5, "other": -0.95}
for g, yl in LANE.items():
    axB.text(-0.015, yl, S.GLAB[g], ha="right", va="center", fontsize=8.4,
             color=S.GCOL[g], fontweight="bold")
seen = {}
for s in sorted(spans, key=lambda s: (LANE[s["group"]], s["x"])):
    g = s["group"]
    yl = LANE[g]
    axB.scatter([s["x"]], [yl], s=42, color=S.GCOL[g], zorder=3,
                edgecolor="white", linewidth=0.5)
    key = (round(yl, 2), round(s["x"], 1))
    st = seen.get(key, 0)
    seen[key] = st + 1
    axB.text(s["x"], yl + 0.14 + st * 0.30, s["code"], rotation=90,
             fontsize=6.6, ha="center", va="bottom", color="#4a545c")
axB.set_yticks([])
S.despine(axB)
axB.set_xticks([b / 10 for b in range(11)])
axB.set_xticklabels([f"{b / 10:.1f}" for b in range(11)], fontsize=7.8)
axB.set_xlabel("start position, discretised into 10 bins", fontsize=9)

# ---- (c) the two real tables ---------------------------------------------------
g2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                      width_ratios=[1.0, 1.05], wspace=0.24)
axP = fig.add_subplot(g2[0])
axT = fig.add_subplot(g2[1])
S.tag(axP, "c", x=-0.155, y=1.025)

posn = pos / pos.sum(1, keepdims=True)
imP = axP.imshow(posn, aspect="auto", cmap=S.ltc_seq, vmin=0,
                 vmax=np.percentile(posn, 99))
axP.set_xticks(range(10))
axP.set_xticklabels([f".{i}" for i in range(10)], fontsize=7.8)
axP.set_yticks(range(n))
axP.set_yticklabels(labels, fontsize=7.6)
for t, f in zip(axP.get_yticklabels(), fam):
    t.set_color(S.GCOL[f])
axP.set_xlabel("position bin", fontsize=9)
axP.set_ylabel("clinician code", fontsize=9)
axP.set_title("P(bin | code)   $\\rightarrow\\ C$", fontsize=10.5, pad=6,
              color=S.INK)
for b in bnd:
    axP.axhline(b - 0.5, color="white", lw=2.2)

trn = tr / np.clip(tr.sum(1, keepdims=True), 1, None)
imT = axT.imshow(trn, aspect="auto", cmap=S.ltc_seq, vmin=0,
                 vmax=np.percentile(trn, 98))
axT.set_xticks(range(n))
axT.set_xticklabels(labels, fontsize=6.6, rotation=90)
axT.set_yticks(range(n))
axT.set_yticklabels(labels, fontsize=6.6)
for t, f in zip(axT.get_yticklabels(), fam):
    t.set_color(S.GCOL[f])
for t, f in zip(axT.get_xticklabels(), fam):
    t.set_color(S.GCOL[f])
axT.set_xlabel("next code", fontsize=9)
axT.set_ylabel("current code", fontsize=9)
axT.set_title("P(next | current)   $\\rightarrow\\ M$", fontsize=10.5, pad=6,
              color=S.INK)
for b in bnd:
    axT.axhline(b - 0.5, color="white", lw=1.6)
    axT.axvline(b - 0.5, color="white", lw=1.6)
for im_, ax_ in [(imP, axP), (imT, axT)]:
    cb = fig.colorbar(im_, ax=ax_, orientation="horizontal", fraction=0.035,
                      pad=0.13 if ax_ is axP else 0.16)
    cb.set_ticks([])
    cb.set_label("rare  \u2192  frequent", fontsize=8)
    cb.outline.set_edgecolor("#c8cdd2")

# ---- (d) the score -------------------------------------------------------------
axD = fig.add_subplot(gs[3])
S.tag(axD, "d", x=-0.075, y=0.92)
sc = F["score"]
C, M, script, z = sc["C_excess"], sc["M_excess"], sc["SCRIPT"], sc["z"]
axD.add_patch(plt.Rectangle((0, 0.28), C, 0.34, fc=S.GCOL["empathy"],
                            ec="white", lw=0.8))
axD.add_patch(plt.Rectangle((C, 0.28), M, 0.34, fc=S.GCOL["advice"],
                            ec="white", lw=0.8))
axD.annotate("", xy=(script, 0.85), xytext=(0, 0.85),
             arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=1.4))
axD.text(script / 2, 0.97, f"SCRIPT = {script:.3f}    ($z$ = {z:.0f})",
         ha="center", fontsize=11, fontweight="bold", color=S.INK)
axD.text(C / 2, 0.10, f"$C$ = {C:.3f}  fixed seats", ha="center", va="top",
         fontsize=9, color=S.GCOL["empathy"], fontweight="bold")
axD.text(C + M / 2, 0.10, f"$M$ = {M:.3f}  chained moves", ha="center",
         va="top", fontsize=9, color=S.GCOL["advice"], fontweight="bold")
axD.text(script + 0.006, 0.45,
         f"all 20 codes \u00d7 10 bins \u00b7 excess over the within-reply "
         f"shuffle \u00b7 {int(sc['n_events']):,} events",
         fontsize=8.6, color=S.MUTE, va="center")
axD.set_xlim(-0.002, 0.24)
axD.set_ylim(-0.45, 1.5)
axD.axis("off")

# ---- flow verbs ----------------------------------------------------------------
def between(ax_top, ax_bot):
    return (ax_top.get_position().y0 + ax_bot.get_position().y1) / 2

S.flow(fig, between(axA, axB) + 0.002,
       "keep one event per highlight:  (code, start bin)", x=0.32)
S.flow(fig, between(axB, axP) - 0.004,
       "pool the events of all 816 Claude replies", x=0.32)
S.flow(fig, axD.get_position().y1 + 0.028,
       "read the structure off the two tables, calibrated against chance",
       x=0.32)

fig.savefig(FIG / "fig_method_faithful.png", dpi=280, bbox_inches="tight")
fig.savefig(FIG / "fig_method_faithful.pdf", bbox_inches="tight")
print("saved", FIG / "fig_method_faithful.png",
      {k: round(float(sc[k]), 3) for k in ["SCRIPT", "C_excess", "M_excess"]})
