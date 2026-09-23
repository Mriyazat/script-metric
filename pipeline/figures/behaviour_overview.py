"""Overview figure of the benchmark behaviour analysis: how much (attribute profile and PCA),
where (position of the four behaviour families), what comes next (advice momentum, closing move),
and for whom (therapist vs. models, user cues). Reads out/tables/behaviour/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

from pipeline.common.style import FAINT, GCOL, INK, MCOL, MUTE, setup
from pipeline.common.paths import ATTRS, GROUPS4, FIG, MODELS, TAB_B

FIG_W_IN, PRINT_W_IN = 8.6, 5.5


def fs(pt):
    """font size in printed points."""
    return pt * FIG_W_IN / PRINT_W_IN


GROUPS = ["empathy_accurate", "empathy_inaccurate", "advice", "questions"]
COL = {"empathy_accurate": GCOL["empathy"], "empathy_inaccurate": "#7fbfcb",
       "advice": GCOL["advice"], "questions": GCOL["questions"]}
LABEL = {"empathy_accurate": "accurate empathy", "empathy_inaccurate": "inaccurate empathy",
         "advice": "advice", "questions": "questions"}
THERAPIST = INK

setup()
plt.rcParams.update({"font.size": fs(6), "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# ----------------------------------------------------------------------------- data
ev = pd.read_csv(TAB_B / "spans_long.csv")
ev = ev[ev.located & ev.position.notna()].rename(columns={"position": "pos"})
ev["group"] = ev.code.map(GROUPS4)
ev = ev.dropna(subset=["group"]).drop_duplicates(["corpus", "row", "model", "group", "pos"])
pos_pooled = pd.read_csv(TAB_B / "position_by_group.csv").set_index("group")
pos_model = pd.read_csv(TAB_B / "position_by_group_model.csv").set_index("model")
stick = pd.read_csv(TAB_B / "advice_momentum.csv").set_index("model")
last = pd.read_csv(TAB_B / "last_span_group_pct.csv").set_index("model")
vul = pd.read_csv(TAB_B / "vulnerability_coupling.csv")
qrate = pd.read_csv(TAB_B / "question_rate_overall.csv").set_index("model").rename(index={"Human": "Human therapist"})
qrate.columns = ["q"]
echo = pd.read_csv(TAB_B / "belief_echo.csv").set_index("speaker").rename(index={"Human": "Human therapist"})

scores = pd.read_csv(TAB_B / "scores_long.csv")
wide = scores.pivot_table(index=["corpus", "row", "model"], columns="attribute", values="value")[ATTRS].dropna()
wide["FIX"] = (wide["FIX"] > 0).astype(float)        # Fix-It read as any-solution, as in the benchmark's risk index
means = wide.groupby(level="model").mean().T.loc[ATTRS, MODELS]
X = (wide - wide.mean()) / wide.std(ddof=0)
sv = np.linalg.svd(X.values, compute_uv=False)
evr = sv ** 2 / (sv ** 2).sum()

# ----------------------------------------------------------------------------- canvas
fig = plt.figure(figsize=(FIG_W_IN, 7.9))
outer = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.22], height_ratios=[1.0, 0.9],
                         left=0.075, right=0.975, top=0.945, bottom=0.065, wspace=0.26, hspace=0.45)


def header(ax, letter, title, dy=0.0):
    """No-op: panel letters and panel titles are given in the caption."""
    return


def clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def colour_model_ticks(ax, axis="y"):
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl in labels:
        if lbl.get_text() in MCOL:
            lbl.set_color(MCOL[lbl.get_text()])
            lbl.set_fontweight("bold")


# ============================================================================= (a) how much
ga = outer[0, 0].subgridspec(2, 1, height_ratios=[1.35, 1], hspace=0.58)
ax_heat = fig.add_subplot(ga[0])
ax_scree = fig.add_subplot(ga[1])

Z = (means.T - means.T.mean()) / (means.T.std(ddof=0) + 1e-9)
ax_heat.imshow(Z.values, cmap="RdBu_r", vmin=-1.9, vmax=1.9, aspect="auto")
ax_heat.set_xticks(range(len(ATTRS)))
ax_heat.set_xticklabels([f"R{i+1}\n{'FIX>0' if a == 'FIX' else a}" for i, a in enumerate(ATTRS)], fontsize=fs(5.2), linespacing=0.95)
ax_heat.set_yticks(range(len(MODELS)))
ax_heat.set_yticklabels(MODELS, fontsize=fs(6))
colour_model_ticks(ax_heat)
ax_heat.tick_params(length=0, pad=2)
for i in range(len(MODELS)):
    for j in range(len(ATTRS)):
        z = Z.iloc[i, j]
        ax_heat.text(j, i, f"{means.iloc[j, i]:.2f}", ha="center", va="center", fontsize=fs(4.5),
                     color="white" if abs(z) > 1.1 else INK)
for s in ax_heat.spines.values():
    s.set_visible(False)
ax_heat.set_xlabel("response attribute (mean clinician score, 0–2; FIX>0: share of replies)", fontsize=fs(5.2), labelpad=3)
header(ax_heat, "a", "how much of each behaviour", dy=0.0)

k = np.arange(1, 6)
ax_scree.bar(k, evr[:5] * 100, color=[INK] + ["#c8cdd2"] * 4, width=0.62)
cum = np.cumsum(evr[:5]) * 100
ax_scree.plot(k, cum, color=GCOL["advice"], marker="o", ms=fs(2.1), lw=1.0, label="cumulative")
for i, c in enumerate(cum):
    ax_scree.text(k[i], c + 4, f"{c:.0f}%", ha="center", fontsize=fs(4.9), color=GCOL["advice"])
ax_scree.text(1, evr[0] * 100 / 2, f"{evr[0]*100:.1f}%", ha="center", va="center", fontsize=fs(4.9),
              color="white", fontweight="bold")
ax_scree.set_xticks(k)
ax_scree.set_xticklabels([f"PC{i}" for i in k], fontsize=fs(5.4))
ax_scree.set_ylim(0, 100)
ax_scree.set_yticks([0, 50, 100])
ax_scree.set_yticklabels(["0", "50", "100%"], fontsize=fs(5))
ax_scree.tick_params(length=2, pad=2)
ax_scree.set_ylabel("variance explained", fontsize=fs(5.4))
ax_scree.legend(loc="lower right", fontsize=fs(5), frameon=False, handlelength=1.4, borderaxespad=0.3)
clean(ax_scree)

# ============================================================================= (b) where
gb = outer[0, 1].subgridspec(2, 1, height_ratios=[0.95, 1.05], hspace=0.08)
ax_seat = fig.add_subplot(gb[0])
ax_dens = fig.add_subplot(gb[1], sharex=ax_seat)

rows = MODELS + ["all models"]
for r, name in enumerate(rows):
    y = len(rows) - 1 - r
    ax_seat.add_patch(FancyBboxPatch((0, y - 0.30), 1, 0.60, boxstyle="round,pad=0,rounding_size=0.08",
                                     fc=FAINT, ec="none"))
    src = pos_pooled["mean"] if name == "all models" else pos_model.loc[name]
    big = name == "all models"
    for g in GROUPS:
        ax_seat.plot(src[g], y, marker="o", ms=fs(4.8 if big else 3.7), color=COL[g], mec="white", mew=0.7, zorder=3)
ax_seat.set_yticks(range(len(rows)))
ax_seat.set_yticklabels(rows[::-1], fontsize=fs(6))
colour_model_ticks(ax_seat)
for lbl in ax_seat.get_yticklabels():
    if lbl.get_text() == "all models":
        lbl.set_fontweight("bold")
ax_seat.set_ylim(-0.6, len(rows) - 0.4)
ax_seat.set_xlim(-0.01, 1.01)
ax_seat.tick_params(axis="x", labelbottom=False, length=0)
ax_seat.tick_params(axis="y", length=0, pad=2)
for s in ax_seat.spines.values():
    s.set_visible(False)
header(ax_seat, "b", "where in the reply", dy=0.0)

xs = np.linspace(0, 1, 400)
for g in GROUPS:
    for m in MODELS:
        d = ev.loc[(ev.group == g) & (ev.model == m), "pos"].values
        if len(d) > 50:
            ax_dens.plot(xs, gaussian_kde(d, 0.22)(xs), color=COL[g], alpha=0.22, lw=0.8)
    d = ev.loc[ev.group == g, "pos"].values
    kde = gaussian_kde(d, 0.22)(xs)
    ax_dens.plot(xs, kde, color=COL[g], lw=2.2, zorder=3)
    ax_dens.fill_between(xs, 0, kde, color=COL[g], alpha=0.07)
ax_dens.set_xlim(-0.01, 1.01)
ax_dens.set_xticks([0, 0.5, 1])
ax_dens.set_xticklabels(["start", "mid", "end"], fontsize=fs(6))
ax_dens.tick_params(axis="x", length=2, pad=2)
ax_dens.set_yticks([])
ax_dens.set_ylabel("span density", fontsize=fs(5.8))
ax_dens.set_xlabel("position of the highlighted span inside the reply", fontsize=fs(6), labelpad=2)
clean(ax_dens, ("top", "right", "left"))
ax_dens.set_ylim(0, ax_dens.get_ylim()[1] * 1.28)
handles = [Line2D([], [], color=COL[g], lw=2.2, label=f"{LABEL[g]}  ({pos_pooled.loc[g,'mean']:.2f})") for g in GROUPS]
handles += [Line2D([], [], color=MUTE, lw=2.2, label="all models pooled"),
            Line2D([], [], color=MUTE, lw=0.8, alpha=0.5, label="each model")]
ax_dens.legend(handles=handles, loc="upper right", fontsize=fs(5.2), frameon=False, handlelength=1.6,
               borderaxespad=0.2, labelspacing=0.3, ncol=2, columnspacing=1.0)

# ============================================================================= (c) what next
gc = outer[1, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.65)
ax_st = fig.add_subplot(gc[0])
ax_last = fig.add_subplot(gc[1])

order = stick.sort_values("momentum_ratio").index.tolist()
for i, m in enumerate(order):
    a, b = stick.loc[m, "p_advice_after_other"], stick.loc[m, "p_advice_after_advice"]
    ax_st.plot([a, b], [i, i], color="#c8cdd2", lw=2.0, zorder=1)
    ax_st.plot(a, i, "o", color="white", mec=MUTE, mew=1.0, ms=fs(3.4), zorder=2)
    ax_st.plot(b, i, "o", color=GCOL["advice"], ms=fs(3.8), zorder=3)
    ax_st.text(b + 0.03, i, f"×{stick.loc[m,'momentum_ratio']:.2f}", va="center", fontsize=fs(5.2),
               color=GCOL["advice"], fontweight="bold")
ax_st.set_yticks(range(len(order)))
ax_st.set_yticklabels(order, fontsize=fs(6))
colour_model_ticks(ax_st)
ax_st.set_xlim(0.22, 0.92)
ax_st.set_xticks([0.3, 0.5, 0.7])
ax_st.set_xticklabels(["0.3", "0.5", "0.7"], fontsize=fs(5.2))
ax_st.tick_params(axis="x", length=2, pad=2)
ax_st.tick_params(axis="y", length=0, pad=2)
ax_st.set_xlabel("P(next span is advice)", fontsize=fs(5.8), labelpad=2)
ax_st.set_ylim(-0.7, len(order) - 0.3)
clean(ax_st)
hs = [Line2D([], [], marker="o", ls="", color="white", mec=MUTE, mew=1.0, ms=fs(3.4), label="after a non-advice span"),
      Line2D([], [], marker="o", ls="", color=GCOL["advice"], ms=fs(3.8), label="after an advice span")]
ax_st.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=fs(5.0), frameon=False,
             handlelength=1.0, borderaxespad=0.0, labelspacing=0.25)
header(ax_st, "c", "what comes next", dy=0.10)

ypos5 = np.arange(len(MODELS))[::-1]
leftv = np.zeros(len(MODELS))
for g in GROUPS:
    v = last.loc[MODELS, g].values
    ax_last.barh(ypos5, v, left=leftv, color=COL[g], height=0.62, edgecolor="white", lw=0.6)
    for y, l, w in zip(ypos5, leftv, v):
        if w >= 14:
            ax_last.text(l + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=fs(4.8),
                         color="white", fontweight="bold")
    leftv = leftv + v
ax_last.set_yticks(ypos5)
ax_last.set_yticklabels(MODELS, fontsize=fs(6))
colour_model_ticks(ax_last)
ax_last.set_xlim(0, 100)
ax_last.set_xticks([0, 50, 100])
ax_last.set_xticklabels(["0", "50", "100%"], fontsize=fs(5.2))
ax_last.tick_params(axis="x", length=2, pad=2)
ax_last.tick_params(axis="y", length=0, pad=2)
ax_last.set_xlabel("family of the last span in the reply", fontsize=fs(5.8), labelpad=2)
ax_last.set_ylim(-0.7, len(MODELS) - 0.3)
clean(ax_last)

# ============================================================================= (d) for whom
gd = outer[1, 1].subgridspec(2, 2, height_ratios=[1.45, 0.7], hspace=0.75, wspace=0.42)
ax_q = fig.add_subplot(gd[0, 0])
ax_e = fig.add_subplot(gd[0, 1], sharey=ax_q)
ax_r = fig.add_subplot(gd[1, :])

speakers = ["Human therapist"] + MODELS
ypos6 = np.arange(len(speakers))[::-1]
colors6 = [THERAPIST] + [MCOL[m] for m in MODELS]


def barh_panel(ax, vals, xlabel, fmt):
    ax.barh(ypos6, vals, color=colors6, height=0.62)
    for y, v in zip(ypos6, vals):
        ax.text(v + max(vals) * 0.03, y, fmt(v), va="center", fontsize=fs(5.0))
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel(xlabel, fontsize=fs(5.6), labelpad=2, linespacing=0.95)
    ax.tick_params(axis="x", labelsize=fs(5.0), length=2, pad=2)
    ax.set_ylim(-0.7, len(speakers) - 0.3)
    clean(ax)


barh_panel(ax_q, [qrate.loc[s, "q"] for s in speakers], "questions per\n100 words", lambda v: f"{v:.2f}")
ax_q.set_yticks(ypos6)
ax_q.set_yticklabels(["therapist"] + MODELS, fontsize=fs(6))
colour_model_ticks(ax_q)
ax_q.get_yticklabels()[0].set_fontweight("bold")
ax_q.tick_params(axis="y", length=0, pad=2)
barh_panel(ax_e, [echo.loc[s, "repeats_unchallenged"] for s in speakers],
           "% of replies repeating the\nuser's negative self-belief", lambda v: f"{v:.0f}%")
ax_e.tick_params(axis="y", labelleft=False, length=0)
fig.canvas.draw()
_yc = ax_st.get_position().y1 + 0.15 * ax_st.get_position().height
_xq = ax_q.get_position().x0

cues = [("user_sensitivity", "explicit safety cue"), ("user_evocative", "affective cue"),
        ("user_underlying", "implicit distress")]
rho = [float(vul.loc[vul.user_cue == key, "spearman"].iloc[0]) for key, _ in cues]
y3 = np.arange(3)[::-1]
ax_r.barh(y3, rho, color=[INK, "#c8cdd2", "#c8cdd2"], height=0.6)
ax_r.axvline(0, color=MUTE, lw=0.6)
for y, r, (_, name) in zip(y3, rho, cues):
    if r < 0:
        ax_r.text(r - 0.006, y, f"{r:+.3f}", va="center", ha="right", fontsize=fs(5.2), color=INK, fontweight="bold")
        ax_r.text(0.008, y, name, va="center", ha="left", fontsize=fs(5.6), color=INK, fontweight="bold")
    else:
        ax_r.text(r + 0.006, y, f"{r:+.3f}   {name}", va="center", ha="left", fontsize=fs(5.4), color=MUTE)
ax_r.set_yticks([])
ax_r.set_xlim(-0.2, 0.2)
ax_r.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
ax_r.set_xticklabels(["−0.15", "−0.1", "−0.05", "0", "0.05", "0.1"], fontsize=fs(5.0))
ax_r.tick_params(axis="x", length=2, pad=2)
ax_r.set_ylim(-0.7, 2.7)
ax_r.set_xlabel("Spearman ρ between the user cue and the advice space in the reply", fontsize=fs(5.6), labelpad=2)
ax_r.tick_params(axis="y", length=0)
clean(ax_r, ("top", "right", "left"))

# ----------------------------------------------------------------------------- write
fig.savefig(FIG / "fig_behaviour_overview.png", dpi=300)
fig.savefig(FIG / "fig_behaviour_overview.pdf")
print("wrote", FIG / "fig_behaviour_overview.pdf")
