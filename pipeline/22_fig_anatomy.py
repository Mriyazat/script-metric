"""Figure 1 (fig_anatomy), three panels, all prose in the caption:
  (a) nine real Claude replies (spans at their positions) + the pooled
      positional density of each behaviour group over ALL Claude replies:
      three separated humps -- the seating chart itself
  (b) the same replies and densities after one within-response shuffle:
      same positions, same mix, but the densities collapse onto one shape
  (c) observed R vs. the 200-draw null on all pooled Claude events; the
      gap is SCRIPT, split into C (choreography) and M (momentum)

Palette and style follow 09_make_figures_explanatory.py exactly.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paths import EVENTS, FIG

import script_metric as sm

G3 = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
      **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
      **{c: "questions" for c in ["QOP", "QCL"]}}
gcolors = {"empathy": "#00798c", "advice": "#d1495b", "questions": "#66a182",
           "other": "#b9b9b9"}
C_COL, M_COL, CHANCE_COL = "#00798c", "#E5A11F", "#d9d9d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E["group"] = E.label.map(G3).fillna("other")
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)

m = "Claude"

# ---- the nine example replies (same selection as the original figure) -------
rng = np.random.default_rng(4)
d = E[(E.model == m) & (E.corpus.isin(["carebench", "hope"]))]
agg = d.groupby("response_id").agg(n=("group", "size"),
        emp=("group", lambda g: (g == "empathy").sum()),
        adv=("group", lambda g: (g == "advice").sum()),
        que=("group", lambda g: (g == "questions").sum()),
        oth=("group", lambda g: (g == "other").sum()))
cands = agg[(agg.n >= 6) & (agg.n <= 9) & (agg.emp >= 2) & (agg.que >= 1)
            & (agg.adv >= 1) & (agg.oth <= 2)].index.to_numpy()
picks = rng.choice(cands, min(9, len(cands)), replace=False)

# ---- pooled positional densities, real and one shuffle draw -----------------
allc = E[E.model == m][["response_id", "group", "position"]].reset_index(drop=True)
rng_d = np.random.default_rng(11)
shuf_groups = allc.groupby("response_id", sort=False).group \
    .transform(lambda g: rng_d.permutation(g.to_numpy()))

def kde(pos, bw=0.045, xs=np.linspace(0, 1, 240)):
    pos = np.asarray(pos)
    dens = np.exp(-0.5 * ((xs[:, None] - pos[None, :]) / bw) ** 2).sum(1)
    return xs, dens / np.trapezoid(dens, xs)

DGROUPS = ["empathy", "advice", "questions"]
dens_real = {g: kde(allc.position[allc.group == g]) for g in DGROUPS}
dens_shuf = {g: kde(allc.position[shuf_groups == g]) for g in DGROUPS}

# ---- panel (c): observed vs 200-draw null on ALL pooled Claude events -------
dc = E[E.model == m][["response_id", "label", "position"]] \
        .sort_values(["response_id", "position"]).reset_index(drop=True)
labels = sorted(dc.label.astype(str).unique())
l2i = {l: i for i, l in enumerate(labels)}
lab = dc.label.astype(str).map(l2i).to_numpy()
n_bins = 10
xbin = np.minimum((dc.position.to_numpy() * n_bins).astype(int), n_bins - 1)
rid = pd.factorize(dc.response_id)[0]
C0, M0, R0 = sm._metrics(lab, xbin, rid, len(labels), n_bins)
rng_null = np.random.default_rng(0)
null = np.array([sm._metrics(sm._shuffle_within(lab, rid, rng_null), xbin, rid,
                             len(labels), n_bins) for _ in range(200)])
muC, muM, muR = null.mean(0)
sdR = null[:, 2].std()
script, z = R0 - muR, (R0 - muR) / sdR
Cx, Mx = C0 - muC, M0 - muM

# ---- layout ------------------------------------------------------------------
fig = plt.figure(figsize=(15.0, 5.3))
outer = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.88], wspace=0.15)
gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                       height_ratios=[2.5, 1.0], hspace=0.10)
gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                       height_ratios=[2.5, 1.0], hspace=0.10)
axA = fig.add_subplot(gsA[0])
axAd = fig.add_subplot(gsA[1], sharex=axA)
axB = fig.add_subplot(gsB[0], sharey=axA)
axBd = fig.add_subplot(gsB[1], sharex=axB, sharey=axAd)
axC = fig.add_subplot(outer[2])

# ---- panels (a)/(b) top: the nine replies ------------------------------------
for ax, mode in [(axA, "real"), (axB, "shuffled")]:
    for yi, ridx in enumerate(picks):
        dd = d[d.response_id == ridx].sort_values("position")
        groups = dd.group.to_numpy()
        if mode == "shuffled":
            groups = rng.permutation(groups)
        ax.hlines(yi, 0, 1, color="#eeeeee", lw=8.5, zorder=1)
        for x, g in zip(dd.position, groups):
            ax.plot([x], [yi], marker="s", ms=10, color=gcolors[g],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(picks) - 0.3)
    ax.set_yticks(range(len(picks)))
    ax.invert_yaxis()
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
axA.set_yticklabels([f"reply {i + 1}" for i in range(len(picks))], fontsize=8.5)
plt.setp(axB.get_yticklabels(), visible=False)
axB.tick_params(axis="y", length=0)

# ---- panels (a)/(b) bottom: pooled positional densities -----------------------
for axd, dens in [(axAd, dens_real), (axBd, dens_shuf)]:
    for g in DGROUPS:
        xs, ys = dens[g]
        axd.fill_between(xs, 0, ys, color=gcolors[g], alpha=0.35, lw=0, zorder=2)
        axd.plot(xs, ys, color=gcolors[g], lw=1.8, zorder=3)
    axd.set_xlim(-0.03, 1.03)
    axd.set_ylim(0, None)
    axd.set_yticks([])
    axd.set_xticks([0, 0.5, 1])
    axd.set_xticklabels(["start", "mid", "end"], fontsize=10)
    axd.spines[["top", "right", "left"]].set_visible(False)
axAd.set_ylabel("all replies, pooled", fontsize=7.5, color="#666666")

# panel tags
for ax, tag in [(axA, "(a)"), (axB, "(b)"), (axC, "(c)")]:
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left", color="#333333")

handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color=gcolors[g], label=g)
           for g in ["empathy", "advice", "questions", "other"]]
axAd.legend(handles=handles, fontsize=9, frameon=False, ncol=4,
            loc="upper center", bbox_to_anchor=(1.05, -0.34))

# ---- panel (c) -----------------------------------------------------------------
xs = np.linspace(muR - 7 * sdR, muR + 7 * sdR, 400)
dens = np.exp(-0.5 * ((xs - muR) / sdR) ** 2)
axC.fill_between(xs, 0, dens * 0.60, color="#c9c9c9",
                 edgecolor="#777777", lw=1.0, zorder=2)
axC.axvline(muR, color="#888888", ls="--", lw=1.1, zorder=1)
axC.axvline(R0, color="#333333", lw=2.2, zorder=4)
axC.text(muR, 0.995, "SCRIPT = 0", ha="center", va="top", fontsize=9,
         fontweight="bold", color="#666666",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
axC.text(muR, 0.645, "chance\n(200 shuffle draws)", ha="center", va="bottom",
         fontsize=8.5, color="#666666",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
axC.text(R0 - 0.002, 0.985, "$R_{\\mathrm{obs}}$" + f" = {R0:.3f}",
         ha="right", va="top", fontsize=9.5, color="#333333")

y_arrow = 0.82
axC.annotate("", xy=(R0, y_arrow), xytext=(muR, y_arrow),
             arrowprops=dict(arrowstyle="<|-|>", color="#333333", lw=1.6))
axC.text((muR + R0) / 2, y_arrow + 0.03,
         f"SCRIPT = {script:.3f}", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#333333")
axC.text((muR + R0) / 2, y_arrow - 0.04,
         f"$z \\approx {z:.0f}$", ha="center", va="top",
         fontsize=9.5, color="#666666")

y_bar, h_bar = 0.30, 0.08
axC.barh(y_bar, Cx, left=muR, height=h_bar, color=C_COL,
         edgecolor="white", zorder=3)
axC.barh(y_bar, Mx, left=muR + Cx, height=h_bar, color=M_COL,
         edgecolor="white", zorder=3)
axC.text(muR + Cx / 2, y_bar - 0.075, f"C = {Cx:.3f}\nchoreography",
         ha="center", va="top", fontsize=8, color=C_COL)
axC.text(muR + Cx + Mx / 2, y_bar - 0.075, f"M = {Mx:.3f}\nmomentum",
         ha="center", va="top", fontsize=8, color=M_COL)

axC.set_xlim(muR - 0.034, R0 + 0.014)
axC.set_ylim(0, 1.0)
axC.set_yticks([])
axC.set_xlabel("structural share $R$", fontsize=10)
axC.spines[["top", "right", "left"]].set_visible(False)

fig.savefig(FIG / "fig_anatomy.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_anatomy.pdf", bbox_inches="tight")
print(f"Claude pooled: R_obs={R0:.4f} null={muR:.4f} SCRIPT={script:.4f} "
      f"z={z:.1f} C_excess={Cx:.4f} M_excess={Mx:.4f}")
print("saved", FIG / "fig_anatomy.png")
