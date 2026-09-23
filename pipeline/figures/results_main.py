#!/usr/bin/env python3
"""The one results figure for 5.1-5.2: two cards, self-explanatory.

(a) The same response, ten times: for the therapist and the five models, the
    behaviour that dominates each seat of the response (columns, start to end)
    at each turn of the conversation (rows, 1-10). Blind LLM layer, multi-turn.
(b) What moves the models: each model's shift of behaviour mix with a user cue,
    as a share of the therapist's shift on the same cue, cues from felt to stated.

Numbers already in Tables 2-3 (C, M, advice shares) are deliberately not repeated."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Patch

from pipeline.common.paths import FIG, MODELS, TAB, DERIVED
from pipeline.external.listening import prep, turn_meta

CREAM, CARD, INK, MUTE, FAINT = "#f6f4ef", "#ffffff", "#26323a", "#7c8790", "#e6e9ec"
EMP, ADV, QUE = "#3d6fae", "#d1495b", "#4f9d6b"
MCOL = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182", "Claude": "#d1495b", "Gemini": "#00798c"}
LAB = {**{m: m for m in MODELS}, "Human": "Therapist"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.edgecolor": "#c8cdd2"})

L = prep(pd.read_csv(DERIVED / "llm_span_events.csv"), turn_meta())
L["g3"] = L.g.map({"emp_acc": "emp", "emp_in": "emp", "advice": "adv", "quest": "que"}).fillna("oth")
L["xb"] = np.minimum((L.position * 10).astype(int), 9)
us = pd.read_csv(TAB / "listening_user_state.csv")
us = us[us.alphabet == "20-code"]

W, H = 11.0, 4.6
fig = plt.figure(figsize=(W, H), facecolor=CREAM)


def card(x0, y0, w, h, letter, title):
    fig.patches.append(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor="#e2e5e8", lw=1, zorder=-5))
    fig.patches.append(Circle((x0 + 0.022, y0 + h - 0.065), 0.011, transform=fig.transFigure,
                              facecolor=INK, edgecolor="none", zorder=2))
    fig.text(x0 + 0.022, y0 + h - 0.065, letter, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=3)
    fig.text(x0 + 0.04, y0 + h - 0.065, title, ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)


# ------------------------------------------------------------------ (a) strips
xa, ya, wa, ha = 0.012, 0.03, 0.615, 0.94
card(xa, ya, wa, ha, "a", "The same response, ten times")
gcol = {"emp": EMP, "adv": ADV, "que": QUE}
ax0 = xa + 0.055; ay0 = ya + 0.17; aw = wa - 0.075; ah = ha - 0.34
strip_w = (aw - 0.008 * 5) / 6
for k, sp in enumerate(["Human"] + MODELS):
    ax = fig.add_axes((ax0 + k * (strip_w + 0.008), ay0, strip_w, ah)); ax.set_zorder(1)
    E = L[(L.model == sp) & (L.turn <= 10)]
    img = np.ones((10, 10, 3))
    for t in range(1, 11):
        for b in range(10):
            cell = E[(E.turn == t) & (E.xb == b)]
            cell = cell[cell.g3 != "oth"]
            if len(cell) < 3:
                img[t - 1, b] = matplotlib.colors.to_rgb("#f1f3f5"); continue
            sh = cell.g3.value_counts(normalize=True)
            g = sh.idxmax(); a = float(np.clip((sh.max() - 0.34) / 0.5, 0.12, 1.0))
            img[t - 1, b] = 1 - a * (1 - np.array(matplotlib.colors.to_rgb(gcol[g])))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks(range(10) if k == 0 else [])
    if k == 0:
        ax.set_yticklabels([str(t) for t in range(1, 11)], fontsize=8.5)
        ax.set_ylabel("turn of the conversation", fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(LAB[sp], fontsize=11, fontweight="bold", color=INK if sp == "Human" else MCOL[sp], pad=5)
    # arrow under each strip: start -> end of the response
    ax.annotate("", xy=(1.0, -0.06), xytext=(0.0, -0.06), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
fig.text(ax0 + aw / 2, ay0 - 0.075, "seat in the response, start \u2192 end", ha="center", va="top", fontsize=9.5, color=MUTE)
# colour key
fig.legend(handles=[Patch(color=EMP, label="empathy"), Patch(color=ADV, label="advice"), Patch(color=QUE, label="questions"),
                    Patch(facecolor="#f1f3f5", edgecolor="#d5d9dd", label="too few spans")],
           loc="lower left", bbox_to_anchor=(xa + 0.045, ya + 0.015), ncol=4, frameon=False, fontsize=9,
           handlelength=1.2, handleheight=0.9, columnspacing=1.3, handletextpad=0.5)
fig.text(xa + wa - 0.025, ya + 0.045, "darker = firmer seat", ha="right", va="center", fontsize=8.5, color=MUTE, style="italic")

# ------------------------------------------------------------------ (b) felt -> stated
xb, yb, wb, hb = xa + wa + 0.014, 0.03, 1 - (xa + wa + 0.014) - 0.012, 0.94
card(xb, yb, wb, hb, "b", "What moves the models")
ax = fig.add_axes((xb + 0.075, yb + 0.2, wb - 0.1, hb - 0.37)); ax.set_zorder(1)
ax.set_facecolor(CARD)
cues = ["affective intensity", "implicit distress", "atypical presentation", "safety cue"]
cue_lab = ["affective\nintensity", "implicit\ndistress", "atypical\npresent.", "explicit\nsafety cue"]
hum = us[us.speaker == "Human"].set_index("cue").excess
wbar = 0.15
for j, sp in enumerate(MODELS):
    d = us[us.speaker == sp].set_index("cue").reindex(cues)
    ax.bar(np.arange(4) + (j - 2) * wbar, 100 * d.excess / hum.reindex(cues), width=wbar * 0.9, color=MCOL[sp], zorder=3, label=sp)
ax.axhline(100, color=INK, lw=1.5, zorder=4)
ax.text(3.45, 103, "therapist's own shift", ha="right", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(cue_lab, fontsize=8)
ax.set_ylabel("shift of the behaviour mix with the cue,\nas % of the therapist's shift", fontsize=9.5)
ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", labelsize=8.5)
ax.grid(axis="y", color=FAINT, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13),
          handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
fig.text(xb + wb / 2, yb + 0.045, "felt by the person  \u2190\u2500\u2500\u2500\u2500\u2500\u2500\u2192  stated by the person",
         ha="center", va="center", fontsize=9, color=MUTE, style="italic")

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_results_main.{ext}", dpi=250, facecolor=CREAM)
print("saved", FIG / "fig_results_main.png")
