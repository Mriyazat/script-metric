"""Case-study figure (fig_case_study): one round of the identification game,
played in full view.
  (a) the probe: five real annotated replies from the held-out corpus
      (carebench), drawn as span timelines with the model name hidden
  (b) the scoring: mean per-event log-likelihood of the probe under each
      enrolled reference profile (enrolled on the other three corpora);
      the probe is Claude's, and Claude wins with a wide margin
  (c) the hard case: the same game on a Gemini probe --- Gemini wins by a
      hair over its behavioural twin Qwen, with everyone else far behind
  (d) the game repeated: pooled confusion matrix over 200 draws of k=5 per
      (model, held-out corpus); diagonal = correct, and the only systematic
      off-diagonal mass is the twin pair

Palette and style: fig_style (ltc palettes, as everywhere in the paper).
"""
import numpy as np
import pandas as pd
import fig_style as S

S.setup()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paths import EVENTS, FIG, MODELS

import script_metric as sm

MCOLORS = S.MCOL
G3 = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
      **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
      **{c: "questions" for c in ["QOP", "QCL"]}}
gcolors = S.GCOL

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E["group"] = E.label.map(G3).fillna("other")
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
CORPORA = sorted(E.corpus.unique())


def build_profile(df):
    labels = sorted(df.label.astype(str).unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab = df.label.astype(str).map(l2i).to_numpy()
    nb = 10
    xbin = np.minimum((df.position.to_numpy() * nb).astype(int), nb - 1)
    rid = pd.factorize(df.response_id)[0]
    pos = np.zeros((len(labels), nb))
    np.add.at(pos, (lab, xbin), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return dict(labels=labels, position=pos.tolist(), transition=tr.tolist(),
                n_bins=nb)


refs = {held: {m: build_profile(E[(E.model == m) & (E.corpus != held)])
               for m in MODELS} for held in CORPORA}

# ---- the two walkthrough probes (held-out corpus: carebench) ----------------
HELD, K = "carebench", 5


def draw_probe(true_m, seed, rich=False):
    pool = E[(E.model == true_m) & (E.corpus == HELD)]
    rids = pool.response_id.unique()
    if rich:  # display probe: draw among well-annotated replies (5--9 events)
        n = pool.groupby("response_id").size()
        rids = n[(n >= 5) & (n <= 9)].index.to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.choice(rids, K, replace=False)
    probe = pool[pool.response_id.isin(picks)]
    scores = {m: sm.profile_loglik(probe, refs[HELD][m]) for m in MODELS}
    return probe, picks, scores


probe1, picks1, scores1 = draw_probe("Claude", 1, rich=True)  # clear win
probe2, picks2, scores2 = draw_probe("Gemini", 1)             # twin near-miss

# ---- the game repeated: pooled confusion matrix ------------------------------
DRAWS = 200
rng = np.random.default_rng(7)
conf = pd.DataFrame(0.0, index=MODELS, columns=MODELS)
for held in CORPORA:
    for true_m in MODELS:
        pool = E[(E.model == true_m) & (E.corpus == held)]
        rids = pool.response_id.unique()
        for _ in range(DRAWS):
            picks = rng.choice(rids, K, replace=False)
            probe = pool[pool.response_id.isin(picks)]
            ll = {m: sm.profile_loglik(probe, refs[held][m]) for m in MODELS}
            conf.loc[true_m, max(ll, key=ll.get)] += 1
conf = conf / conf.sum(axis=1).values[:, None]
print("pooled confusion:\n", conf.round(3))
print("mean diag:", np.diag(conf).mean().round(3))

# ---- layout ------------------------------------------------------------------
fig = plt.figure(figsize=(14.6, 4.6))
gs = gridspec.GridSpec(1, 4, width_ratios=[1.22, 0.85, 0.85, 0.92],
                       wspace=0.28, left=0.045, right=0.985,
                       top=0.86, bottom=0.17)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])
axD = fig.add_subplot(gs[3])

# ---- (a) the probe: five replies, author hidden --------------------------------
for yi, ridx in enumerate(picks1):
    dd = probe1[probe1.response_id == ridx].sort_values("position")
    axA.hlines(yi, 0, 1, color=S.FAINT, lw=10, zorder=1)
    for x, g in zip(dd.position, dd.group):
        axA.scatter([x], [yi], s=88, color=gcolors[g], zorder=3,
                    edgecolor="white", linewidth=0.9)
axA.text(1.0, -0.72, "author hidden \u2014 who wrote these?", ha="right",
         fontsize=9.3, color=S.INK, style="italic", fontweight="bold")
axA.set_xlim(-0.03, 1.03)
axA.set_ylim(-1.05, K - 0.2)
axA.set_yticks(range(K))
axA.set_yticklabels([f"reply {i + 1}" for i in range(K)], fontsize=9)
axA.invert_yaxis()
axA.set_xticks([0, 0.5, 1])
axA.set_xticklabels(["start", "mid-reply", "end"], fontsize=9)
S.despine(axA)
axA.tick_params(axis="y", length=0)
axA.set_xlabel(f"$k = 5$ annotated replies from an unseen corpus "
               f"({len(probe1)} events)", fontsize=9.3)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=gcolors[g],
                      label=S.GLAB[g])
           for g in ["empathy", "advice", "questions", "other"]]
axA.legend(handles=handles, fontsize=8.5, frameon=False, ncol=4,
           loc="lower center", bbox_to_anchor=(0.5, -0.42),
           handletextpad=0.12, columnspacing=0.8)

# ---- (b)/(c) the lineup: who explains the probe best ---------------------------
def score_panel(ax, scores, true_m, xlab):
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    xmin = min(scores.values()) - 0.32
    xmax = max(scores.values()) + 0.40
    for yi, (m, v) in enumerate(ranked):
        win = yi == 0
        ax.barh(yi, v - xmin, left=xmin, height=0.64, color=MCOLORS[m],
                alpha=0.95 if win else 0.30, zorder=3)
        ax.text(xmin + 0.03, yi, m, ha="left", va="center", fontsize=9.2,
                fontweight="bold" if win else "normal",
                color="white" if win else "#4a545c", zorder=4)
        ax.text(v + 0.04, yi, f"{v:.2f}", ha="left", va="center",
                fontsize=8, color=S.MUTE)
    # margin bracket between the winner's and the runner-up's bar ends
    v1, v2 = ranked[0][1], ranked[1][1]
    ax.plot([v1, v1], [-0.55, 0], color=S.INK, lw=0.9)
    ax.plot([v2, v2], [-0.55, 1], color=S.INK, lw=0.9)
    ax.annotate("", xy=(v1, -0.5), xytext=(v2, -0.5),
                arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=1.1))
    margin = v1 - v2
    ok = ranked[0][0] == true_m
    ax.text(0.0, 1.03, ("\u2713  " if ok else "\u2717  ") + ranked[0][0],
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="#1b8a45" if ok else "#c1393f", va="bottom", ha="left")
    ax.text(0.42, 1.045, f"\u2014 by {margin:.3f}", transform=ax.transAxes,
            fontsize=9, color=S.INK, va="bottom", ha="left")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.0, len(ranked) - 0.25)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel(xlab, fontsize=9.2)
    S.despine(ax)
    ax.tick_params(axis="x", labelsize=8)


score_panel(axB, scores1, "Claude",
            "how well each enrolled routine explains\nthe probe (mean log-likelihood / event)")
score_panel(axC, scores2, "Gemini",
            "the hard case: a Gemini probe\nagainst its behavioural twin")

# ---- (d) the game repeated: confusion matrix -----------------------------------
im = axD.imshow(conf.values, cmap=S.ltc_seq, vmin=0, vmax=1.0)
for i, tm in enumerate(MODELS):
    for j, pm in enumerate(MODELS):
        v = conf.values[i, j]
        axD.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                 ha="center", va="center", fontsize=8.6,
                 fontweight="bold" if i == j else "normal",
                 color="white" if v > 0.45 else S.INK)
axD.set_xticks(range(5))
axD.set_yticks(range(5))
axD.set_xticklabels(MODELS, fontsize=8.2, rotation=45, ha="right")
axD.set_yticklabels(MODELS, fontsize=8.2)
for t, m in zip(axD.get_xticklabels(), MODELS):
    t.set_color(MCOLORS[m])
    t.set_fontweight("bold")
for t, m in zip(axD.get_yticklabels(), MODELS):
    t.set_color(MCOLORS[m])
    t.set_fontweight("bold")
axD.set_xlabel("identified as", fontsize=9.3)
axD.set_ylabel("true model", fontsize=9.3)
iq, ig = MODELS.index("Qwen"), MODELS.index("Gemini")
for (i, j) in [(iq, ig), (ig, iq)]:
    axD.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                edgecolor="#b8860b", lw=2.2, zorder=4))
axD.text(1.0, 1.03, "the twins, boxed", transform=axD.transAxes, fontsize=8.6,
         color="#8a6d00", style="italic", ha="right", va="bottom")

for ax, letter in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
    S.tag(ax, letter, x=-0.02 if ax is not axD else -0.38, y=1.10)

fig.savefig(FIG / "fig_case_study.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_case_study.pdf", bbox_inches="tight")
print("saved", FIG / "fig_case_study.png")
print("case 1 (Claude):", {m: round(v, 3) for m, v in scores1.items()})
print("case 2 (Gemini):", {m: round(v, 3) for m, v in scores2.items()})
