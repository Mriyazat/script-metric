#!/usr/bin/env python3
"""Closing results figure (5.4), three cards in the style of the results figure.

(a) Naming the model from k held-out responses: full profile against the
    behaviour-mix-only and probe-re-dealt baselines, the held-out-annotator
    transfer, and the label-permutation control.
(b) One estimator, many schemes: SCRIPT for every system of every corpus in the
    paper's ruler, from the five Testbed-1 models down to the random-label control.
(c) Certifying a blueprint: SCRIPT-Seq wealth as annotated responses arrive, one
    stream per model and a null stream; rejection at wealth 1/alpha.
"""
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from pipeline.common.paths import EVENTS, FIG, MODELS, OUT, REF, TAB
from scriptmetric import betting as sb

CREAM, CARD, INK, MUTE, FAINT = "#f6f4ef", "#ffffff", "#26323a", "#7c8790", "#e6e9ec"
MCOL = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182", "Claude": "#d1495b", "Gemini": "#00798c"}
HUMAN, SYSGREY, ACC = "#26323a", "#a3a9af", "#2a9d8f"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.edgecolor": "#c8cdd2"})

W, H = 12.6, 4.4
fig = plt.figure(figsize=(W, H), facecolor=CREAM)


def card(x0, y0, w, h, letter, title, inset=0.0, bottom=0.16):
    fig.patches.append(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor="#e2e5e8", lw=1, zorder=-5))
    fig.text(x0 + 0.022, y0 + h - 0.07, letter, ha="center", va="center", fontsize=9.5, fontweight="bold",
             color="white", zorder=3, bbox=dict(boxstyle="circle,pad=0.32", facecolor=INK, edgecolor="none"))
    fig.text(x0 + 0.046, y0 + h - 0.07, title, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    ax = fig.add_axes((x0 + 0.055 + inset, y0 + bottom, w - 0.075 - inset, h - 0.17 - bottom))
    ax.set_zorder(1); ax.set_facecolor(CARD)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    return ax


GAP, MX = 0.007, 0.008
wA, wB, wC = 0.30, 0.37, 0.30
wA, wB, wC = (w * (1 - 2 * MX - 2 * GAP) / (wA + wB + wC) for w in (wA, wB, wC))
xA = MX; xB = xA + wA + GAP; xC = xB + wB + GAP
Y0, HH = 0.03, 0.94

# ------------------------------------------------------------------ (a) naming the model
ax = card(xA, Y0, wA, HH, "a", "Naming the model")
base = pd.read_csv(TAB / "identification_baselines.csv")
curves = pd.read_csv(TAB / "identification_curves.csv")
ks = [1, 5, 20]
def acc(mode): return [float(base[(base["mode"] == mode) & (base.k == k)].accuracy.iloc[0]) for k in ks]
def cur(setting): return [float(curves[(curves.setting == setting) & (curves.k == k)].accuracy.iloc[0]) for k in ks]
ax.plot(ks, acc("full"), "-o", color=ACC, lw=2.6, ms=6, label="full profile, held-out corpus", zorder=5)
ax.plot(ks, acc("mix"), "-o", color="#d1495b", lw=1.8, ms=5, label="behaviour mix only")
ax.plot(ks, acc("redealt"), "--o", color="#8e5bb5", lw=1.8, ms=5, label="full profile, probe re-dealt")
ax.plot(ks, cur("held-out annotator"), "-o", color=SYSGREY, lw=1.8, ms=5, label="full profile, annotator held out")
ax.plot(ks, cur("permutation null"), ":", color=INK, lw=1.4, label="label-permutation control")
ax.axhline(0.2, color="#c8cdd2", lw=1, ls="--"); ax.text(20.4, 0.205, "chance", fontsize=8, color=MUTE, ha="right", va="bottom")
ax.set_xticks(ks); ax.set_xlim(0, 21); ax.set_ylim(0.1, 1.05); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_xlabel("annotated responses in the probe, $k$", fontsize=9.5); ax.set_ylabel("rank-1 accuracy (5 models)", fontsize=9.5)
ax.grid(axis="y", color=FAINT, lw=0.8)
ax.legend(fontsize=7.6, frameon=False, loc="upper left", handlelength=1.8, borderaxespad=0.1)

# ------------------------------------------------------------------ (b) one estimator, many schemes
ax = card(xB, Y0, wB, HH, "b", "One estimator, many schemes", inset=0.042, bottom=0.215)
val = pd.read_csv(REF / "validation_results.csv").set_index("system")
mc = pd.read_csv(TAB / "matched_ceiling.csv").set_index("corpus")
pp = pd.read_csv(TAB / "anchor_propaganda.csv")
control = json.load(open(OUT / "verify_report.json"))["V3"]["SCRIPT"]
rows = [
    ("Testbed 1", [(val.loc[m, "SCRIPT"], MCOL[m]) for m in MODELS], False),
    ("AnnoMI", [(float(mc.loc["AnnoMI human counsellors", "SCRIPT"]), HUMAN)], True),
    ("News", [(float(pp[pp.layer == "human experts"].SCRIPT.iloc[0]), HUMAN)], True),
    ("Data-to-text", [(v, "#66a182") for v in pd.read_csv(TAB / "anchor_d2t.csv").SCRIPT], False),
    ("RAGTruth", [(v, "#7c9fb0") for v in pd.read_csv(TAB / "anchor_ragtruth.csv").SCRIPT], False),
    ("WMT24", [(v, SYSGREY) for v in pd.read_csv(TAB / "anchor_wmt24.csv").SCRIPT], False),
    ("Shuffled labels", [(control, "#888888")], False),
]
for y, (lab, pts, human) in enumerate(rows):
    vals = [v for v, _ in pts]
    core = [v for v in vals if v < 0.15]          # the WMT24 outlier (140 events) is drawn hollow, outside the bar
    if len(core) > 1:
        ax.plot([min(core), max(core)], [y, y], color="#dfe3e6", lw=8, solid_capstyle="round", zorder=1)
    for v, col in pts:
        outlier = v >= 0.15 and not human and col == SYSGREY
        ax.scatter(v, y, s=85 if human else 52, color="white" if outlier else col, zorder=3,
                   edgecolor=col if outlier else "white", lw=1.2 if outlier else 0.7, marker="D" if human else "o")
ax.axvline(0, color="#888888", ls="--", lw=1, zorder=0)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
for t, (_, _, human) in zip(ax.get_yticklabels(), rows):
    if human:
        t.set_fontweight("bold"); t.set_color(INK)
ax.set_ylim(len(rows) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.04, 0.215); ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
ax.set_xlabel("SCRIPT (excess over shuffle)", fontsize=9.5)
ax.spines["left"].set_visible(False)
ax.grid(axis="y", color=FAINT, lw=0.6, zorder=0)
fig.legend(handles=[Line2D([], [], marker="D", ls="", ms=6.5, color=HUMAN, label="human writers"),
                    Line2D([], [], marker="o", ls="", ms=6, color=SYSGREY, label="LLM systems"),
                    Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=SYSGREY, mew=1.2,
                           label="one system, 140 events")],
           loc="lower center", bbox_to_anchor=(xB + wB / 2, Y0 + 0.015), ncol=3, fontsize=8.5, frameon=False,
           handletextpad=0.3, columnspacing=1.4)

# ------------------------------------------------------------------ (c) certifying a blueprint
ax = card(xC, Y0, wC, HH, "c", "Certifying a blueprint")
E = pd.read_csv(EVENTS); E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
ALPHA, MAX_T = 0.05, 120
for m in MODELS:
    d = E[E.model == m][["response_id", "label", "position"]]
    r = sb.test_structure(d, alpha=ALPHA, seed=7, max_t=MAX_T)
    ax.plot(np.arange(1, len(r["trace"]) + 1), r["trace"], color=MCOL[m], lw=1.8, label=m)
rng = np.random.default_rng(7)
nul = E[E.model == "Claude"][["response_id", "label", "position"]].copy()
nul["label"] = nul.groupby("response_id").label.transform(lambda s: rng.permutation(s.to_numpy()))
r0 = sb.test_structure(nul, alpha=ALPHA, seed=7, max_t=MAX_T)
ax.plot(np.arange(1, len(r0["trace"]) + 1), r0["trace"], color="#888888", lw=2.0, ls="--", label="labels re-dealt (null)", zorder=5)
ax.axhline(1 / ALPHA, color=INK, lw=1.1, ls=":")
ax.text(MAX_T - 2, 1 / ALPHA * 1.25, r"reject at wealth $1/\alpha = 20$", fontsize=8, color=INK, ha="right", va="bottom")
ax.set_yscale("log"); ax.set_ylim(0.3, 1e6); ax.set_xlim(0, MAX_T)
ax.set_xlabel("annotated responses seen", fontsize=9.5); ax.set_ylabel("bettor's wealth (log)", fontsize=9.5)
ax.legend(fontsize=7.6, frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=0.9, borderaxespad=0.1)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_results_closing.{ext}", dpi=250, facecolor=CREAM)
print("saved", FIG / "fig_results_closing.png")
