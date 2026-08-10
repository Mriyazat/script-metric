import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import EVENTS, FIG, MODELS, REF, TAB

import script_metric as sm

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}
G3 = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
      **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
      **{c: "questions" for c in ["QOP", "QCL"]}}
gcolors = {"empathy": "#00798c", "advice": "#d1495b", "questions": "#66a182",
           "other": "#b9b9b9"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E["group"] = E.label.map(G3).fillna("other")
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)

# ============ FIG A: anatomy — real replies vs their own shuffle null
rng = np.random.default_rng(4)
m = "Claude"
d = E[(E.model == m) & (E.corpus.isin(["carebench", "hope"]))]
agg = d.groupby("response_id").agg(n=("group", "size"),
        emp=("group", lambda g: (g == "empathy").sum()),
        adv=("group", lambda g: (g == "advice").sum()),
        que=("group", lambda g: (g == "questions").sum()),
        oth=("group", lambda g: (g == "other").sum()))
cands = agg[(agg.n >= 6) & (agg.n <= 9) & (agg.emp >= 2) & (agg.que >= 1)
            & (agg.adv >= 1) & (agg.oth <= 2)].index.to_numpy()
picks = rng.choice(cands, min(9, len(cands)), replace=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
for ax, mode in zip(axes, ["real", "shuffled"]):
    for yi, ridx in enumerate(picks):
        dd = d[d.response_id == ridx].sort_values("position")
        groups = dd.group.to_numpy()
        if mode == "shuffled":
            groups = rng.permutation(groups)
        ax.hlines(yi, 0, 1, color="#eeeeee", lw=9, zorder=1)
        for x, g in zip(dd.position, groups):
            ax.plot([x], [yi], marker="s", ms=11, color=gcolors[g],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(picks) - 0.3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["start", "mid", "end"], fontsize=10)
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels([f"reply {i + 1}" for i in range(len(picks))], fontsize=8.5)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].set_title("REAL — nine Claude replies, nine different people\n"
                  "same seating every time", fontsize=11.5)
axes[1].set_title("SHUFFLE NULL — same replies, same positions,\nlabels randomly re-seated",
                  fontsize=11.5)
handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color=gcolors[g], label=g)
           for g in ["empathy", "advice", "questions", "other"]]
axes[0].legend(handles=handles, fontsize=9, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.suptitle("What the metric sees — SCRIPT is the measured distance between these two worlds",
             fontsize=13.5, y=1.03)
fig.savefig(FIG / "fig_anatomy.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_anatomy.pdf", bbox_inches="tight")
plt.close(fig)

# ============ FIG B: where the bits go — decomposition of H(L)
rows = {}
for mm in MODELS:
    dd = E[E.model == mm][["response_id", "label", "position"]]
    res, _ = sm.compute(dd.sort_values(["response_id", "position"]).reset_index(drop=True),
                        n_bins=10, n_shuffles=40, seed=0)
    rows[mm] = res
fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = np.arange(len(MODELS))[::-1]
for yi, mm in zip(y, MODELS):
    r = rows[mm]
    chance = r["R_null"]
    c = r["C_excess"]
    mo = r["M_excess"]
    rest = 1 - chance - c - mo
    left = 0
    for val, col, lab in [(chance, "#d9d9d9", "chance (small-sample + composition)"),
                          (c, "#00798c", "C — choreography"),
                          (mo, "#E5A11F", "M — momentum"),
                          (rest, "#f4efe9", "content / unexplained")]:
        ax.barh(yi, val, left=left, color=col, edgecolor="white",
                label=lab if mm == MODELS[0] else None)
        left += val
    ax.text(rows[mm]["R_null"] + (c + mo) / 2, yi, f"script\n{c + mo:.3f}",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(MODELS, fontsize=11)
for yi, mm in zip(y, MODELS):
    ax.get_yticklabels()[list(y).index(yi)].set_color(MCOLORS[mm])
ax.set_xlim(0, 1)
ax.set_xlabel("share of label uncertainty H(L)", fontsize=10)
ax.legend(fontsize=8.2, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=2)
ax.set_title("Where the bits go — every model's guessing game, decomposed\n"
             "the grey is what the null removes; the coloured slice is the SCRIPT score; the rest is content",
             fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIG / "fig_bits.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_bits.pdf", bbox_inches="tight")
plt.close(fig)

# ============ FIG C: human anchor + expert-judgment link
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
# left: shared rule-alphabet speakers including the human therapist
# (precomputed under a 3-label surface alphabet applied identically to all
# speakers on identical prompts; built by 12_therapist_baseline.py)
sp = pd.read_csv(REF / "therapist_baseline.csv")
sp = sp.sort_values("SCRIPT")
ax = axes[0]
cols = [("#555555" if s == "Therapist" else MCOLORS.get(s, "#999")) for s in sp.speaker]
bars = ax.barh(sp.speaker, sp.SCRIPT, color=cols, edgecolor="white")
for b, v, s in zip(bars, sp.SCRIPT, sp.speaker):
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
ax.get_yticklabels()[list(sp.speaker).index("Therapist")].set_fontweight("bold")
ax.axvline(sp[sp.speaker == "Therapist"].SCRIPT.iloc[0], color="#555555", ls="--", lw=1)
ax.set_xlabel("SCRIPT (shared 3-label rule alphabet, same prompts)")
ax.set_title("C1 — The human anchor: on identical prompts, the\ntherapist is the least scripted speaker",
             fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
# right: the expert-judgment link, before and after matching stratum size.
# EMP is the clinician empathy-accuracy rating (0 = inaccurate, 2 = accurate).
# The accurate stratum is much the rarer one and SCRIPT is downward-biased at
# small n, so the raw dumbbells (ghosted) are confounded; the filled markers
# re-score both strata at a common response count (script 15).
qm = pd.read_csv(TAB / "quality_matched.csv")
qm = qm[qm.axis == "empathy accuracy"].set_index("model")
ax = axes[1]
yy = np.arange(len(MODELS))[::-1]
for yi, mm in zip(yy, MODELS):
    r = qm.loc[mm]
    # ghosted: the comparison as scored, at unequal n
    ax.plot([r.SCRIPT_bad, r.SCRIPT_good], [yi + 0.20] * 2, color="#bbbbbb",
            lw=1.2, ls=":", zorder=1)
    ax.scatter([r.SCRIPT_bad, r.SCRIPT_good], [yi + 0.20] * 2, s=26,
               facecolor="white", edgecolor="#bbbbbb", linewidth=1.0, zorder=2)
    # matched: same two strata at a common response count, with sampling spread
    ax.plot([r.SCRIPT_bad_matched, r.SCRIPT_good_matched], [yi - 0.06] * 2,
            color=MCOLORS[mm], lw=2, alpha=0.55, zorder=2)
    ax.errorbar([r.SCRIPT_bad_matched, r.SCRIPT_good_matched], [yi - 0.06] * 2,
                xerr=[r.sd_bad, r.sd_good], fmt="none", ecolor=MCOLORS[mm],
                elinewidth=1.3, capsize=3, alpha=0.9, zorder=3)
    ax.scatter([r.SCRIPT_bad_matched], [yi - 0.06], s=90, color=MCOLORS[mm],
               marker="o", zorder=4)
    ax.scatter([r.SCRIPT_good_matched], [yi - 0.06], s=130, color=MCOLORS[mm],
               marker="*", zorder=4)
    if r.verdict == "survives":
        ax.text(min(r.SCRIPT_good_matched, r.SCRIPT_bad_matched) - 0.006,
                yi - 0.06, "resolved", ha="right", va="center", fontsize=7.5,
                style="italic", color=MCOLORS[mm])
ax.set_yticks(yy)
ax.set_yticklabels(MODELS, fontsize=10)
for t, mm in zip(ax.get_yticklabels(), MODELS):
    t.set_color(MCOLORS[mm])
ax.set_ylim(-1.15, len(MODELS) - 0.35)
ax.scatter([], [], marker="o", color="k", label="empathy rated inaccurate (EMP=0)")
ax.scatter([], [], marker="*", s=130, color="k", label="empathy rated accurate (EMP=2)")
ax.plot([], [], ls=":", color="#bbbbbb", marker="o", markerfacecolor="white",
        markeredgecolor="#bbbbbb", label="as scored, at unequal $n$")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_xlabel("SCRIPT within stratum (error bars: resampling at matched $n$)")
ax.set_title("C2 — Matched for sample size, the quality gap mostly goes:\n"
             "only Llama's is resolved; GPT and Gemini invert",
             fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIG / "fig_human_quality.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_human_quality.pdf", bbox_inches="tight")
plt.close(fig)

# ============ FIG D: the evidence ladder — score vs sample size
sc = pd.read_csv(TAB / "sample_size_curve.csv")
fig, ax1 = plt.subplots(figsize=(9.5, 4.6))
ax2 = ax1.twinx()
for mm, ls in [("Claude", "-"), ("Llama", "--")]:
    d = sc[sc.model == mm]
    ax1.errorbar(d.n_resp, d.SCRIPT_mean, yerr=d.SCRIPT_sd, fmt="o" + ls,
                 color=MCOLORS[mm], lw=1.8, ms=5, capsize=3, label=f"{mm} — SCRIPT")
    ax2.plot(d.n_resp, d.z_mean, ls, color=MCOLORS[mm], alpha=0.35, lw=1.2)
ax1.set_xscale("log")
ax1.set_xticks([25, 50, 100, 200, 400, 800])
ax1.set_xticklabels([25, 50, 100, 200, 400, 800])
ax1.set_xlabel("annotated responses (log scale)")
ax1.set_ylabel("SCRIPT (mean ± SD over 10 draws)")
ax2.set_ylabel("z vs null (faint lines)", color="#888888")
ax2.tick_params(axis="y", colors="#888888")
for x0, x1, lab, col in [(25, 50, "noise", "#fbeaea"), (50, 150, "whisper", "#fdf3e3"),
                         (150, 450, "clear", "#eef5ec"), (450, 900, "proof", "#e7eef6")]:
    ax1.axvspan(x0, x1, color=col, zorder=0)
    ax1.text(np.sqrt(x0 * x1), 0.118, lab, ha="center", fontsize=9, color="#666666")
ax1.set_ylim(-0.045, 0.128)
ax1.legend(fontsize=9, frameon=False, loc="lower right")
ax1.set_title("The evidence ladder — the same computation at every scale\n"
              "the point value is a LOWER bound at small n; below ~1,000 events, quote z, not the value",
              fontsize=12)
ax1.spines[["top"]].set_visible(False)
fig.savefig(FIG / "fig_evidence_ladder.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_evidence_ladder.pdf", bbox_inches="tight")
plt.close(fig)
print("done")
