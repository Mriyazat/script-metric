"""Appendix figures of the benchmark behaviour analysis, one per level:
attributes and flags, the span layer, turn dynamics, and repeated language.
Reads out/tables/behaviour/ and the multi-turn corpora."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pipeline.common import benchmark as bm
from pipeline.common.style import FAINT, GCOL, INK, MCOL, MUTE, ltc_seq, setup
from pipeline.common.paths import ATTRS, FIG, MODELS, TAB_B

FIG_W_IN, PRINT_W_IN = 8.6, 5.5


def fs(pt):
    """font size in printed points (figure is scaled to PRINT_W_IN in the paper)."""
    return pt * FIG_W_IN / PRINT_W_IN


setup()
plt.rcParams.update({"font.size": fs(6), "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

ANAME = {"S": "Sensitivity", "AUR": "Assumes user\naccurate", "TD": "Tentative /\ndirective",
         "FIX": "Fix-It", "RT": "Recommendation\ntype", "TN": "Topic shift", "QOC": "Question type",
         "LM": "Language\nmatching", "ME": "Minimal\nencouragers", "EMP": "Empathy"}
LEVEL_COL = ["#e3e7ea", "#94D2BD", "#0A9396"]
FLAGS = [("yn_decisive", "Directive"), ("yn_assumes", "Assumes user\nexperience"),
         ("yn_introduces", "Introduces\nnew content"), ("yn_harmful", "Potentially harmful\nvalidation"),
         ("yn_incoherent", "Incoherent")]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d1495b", "#ffffff", "#00798c"])


def clean(ax, hide=("top", "right")):
    for s in hide:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def tag(ax, letter, title=None, x=-0.02, y=1.06, pad=6):
    """No-op: panel letters and panel titles are given in the caption."""
    return


def figtag(fig, letter, title, x, y):
    """No-op: panel letters and panel titles are given in the caption."""
    return


def save(fig, name):
    out_png, out_pdf = FIG / f"{name}.png", FIG / f"{name}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("wrote", out_pdf.name)


# =============================================================================== Level 1
scores = pd.read_csv(TAB_B / "scores_long.csv")
wide = scores.pivot_table(index=["corpus", "row", "model"], columns="attribute", values="value")[ATTRS].dropna()
wide_bin = wide.copy()
wide_bin["FIX"] = (wide_bin["FIX"] > 0).astype(float)            # 1[FIX>0], as in the benchmark ARI

flags = pd.read_csv(TAB_B / "flags_by_model.csv").set_index(["model", "flag"])


def level1():
    fig = plt.figure(figsize=(FIG_W_IN, 6.8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.3,
                  left=0.075, right=0.985, top=0.9, bottom=0.075)

    # ---- (a) score distributions: 10 mini panels, 5 stacked bars each
    gsa = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0], hspace=0.45, wspace=0.28)
    first = None
    for i, a in enumerate(ATTRS):
        ax = fig.add_subplot(gsa[i // 5, i % 5])
        first = first or ax
        col = wide_bin[a] if a == "FIX" else wide[a]
        levels = [0, 1] if a in ("S", "FIX") else [0, 1, 2]
        for j, m in enumerate(MODELS):
            v = col.xs(m, level="model")
            left = 0.0
            for lv in levels:
                share = float((v == lv).mean()) * 100
                ax.barh(j, share, left=left, height=0.72, color=LEVEL_COL[lv], edgecolor="white", linewidth=0.5)
                if share >= 14:
                    ax.text(left + share / 2, j, f"{share:.0f}", ha="center", va="center", fontsize=fs(4.6),
                            color=INK if lv == 0 else "white")
                left += share
        ax.set_yticks(range(5))
        ax.set_yticklabels(MODELS if i % 5 == 0 else [""] * 5, fontsize=fs(5.2))
        for t, m in zip(ax.get_yticklabels(), MODELS):
            t.set_color(MCOL[m])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"] if i >= 5 else ["", "", ""], fontsize=fs(4.8))
        ax.tick_params(axis="y", length=0)
        clean(ax, ("top", "right", "left"))
        ax.set_title(f"R{i + 1}: {a}" + (" (>0)" if a == "FIX" else ""), fontsize=fs(5.6), color=INK, pad=3, loc="left")
    figtag(fig, "a", "", 0.03, 0.935)
    fig.legend(handles=[Patch(color=LEVEL_COL[k], label=f"score = {k}") for k in range(3)],
               loc="lower right", bbox_to_anchor=(0.985, 0.935), ncol=3, frameon=False, fontsize=fs(5.4),
               handlelength=1.2, columnspacing=1.0)

    # ---- bottom row
    gsb = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[1.35, 0.75, 1.1], wspace=0.5)

    # (b) Spearman correlation among the ten attributes
    ax = fig.add_subplot(gsb[0])
    rho = wide_bin.corr(method="spearman").loc[ATTRS, ATTRS].values
    im = ax.imshow(rho, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            v = rho[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-", "−"), ha="center", va="center",
                    fontsize=fs(4.4), color=INK if abs(v) < 0.6 else "white",
                    fontweight="bold" if abs(v) >= 0.5 else "normal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(ATTRS, fontsize=fs(5), rotation=90); ax.set_yticklabels(ATTRS, fontsize=fs(5))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=fs(4.6), length=2)
    cb.outline.set_visible(False)
    tag(ax, "b", "Spearman ρ, ten attributes", x=-0.12)

    # (c) PCA scree
    ax = fig.add_subplot(gsb[1])
    X = ((wide_bin - wide_bin.mean()) / wide_bin.std(ddof=0)).values
    _, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    evr = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(evr)
    k = np.arange(1, 11)
    ax.bar(k, evr * 100, color="#94D2BD", width=0.7, label="component")
    ax.plot(k, cum * 100, color=INK, lw=1.2, marker="o", ms=3, label="cumulative")
    ax.axhline(cum[4] * 100, color=MUTE, lw=0.6, ls=":")
    ax.text(1.6, 21, f"PC1 = {evr[0] * 100:.1f}%", fontsize=fs(5.2), color=INK, ha="left", va="center")
    ax.text(5.6, cum[4] * 100 - 9, f"5 components\n= {cum[4] * 100:.0f}%", fontsize=fs(5.2), color=INK, ha="left", va="top")
    ax.set_xticks(k); ax.set_xticklabels([str(i) for i in k], fontsize=fs(5))
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=fs(5))
    ax.set_ylim(0, 105)
    ax.set_xlabel("principal component", fontsize=fs(5.4), labelpad=2)
    ax.set_ylabel("explained variance", fontsize=fs(5.4), labelpad=2)
    ax.text(10.1, 85, "cumulative", fontsize=fs(5), color=INK, ha="right", va="top")
    ax.text(4.4, 12, "per component", fontsize=fs(5), color="#3f8f7f", ha="left", va="bottom")
    clean(ax)
    tag(ax, "c", "PCA scree", x=-0.18)

    # (d) flag firing rates with Wilson CIs
    ax = fig.add_subplot(gsb[2])
    off = np.linspace(-0.3, 0.3, 5)
    for i, (f, lab) in enumerate(FLAGS):
        ax.axhspan(i - 0.5, i + 0.5, color=FAINT if i % 2 == 0 else "white", zorder=0, lw=0)
        for j, m in enumerate(MODELS):
            r = flags.loc[(m, f)]
            ax.plot([r.ci_lo * 100, r.ci_hi * 100], [i + off[j]] * 2, color=MCOL[m], lw=1.0, solid_capstyle="round")
            ax.plot(r.rate * 100, i + off[j], "o", color=MCOL[m], ms=3.2, mec="white", mew=0.4)
    ax.set_yticks(range(len(FLAGS))); ax.set_yticklabels([l for _, l in FLAGS], fontsize=fs(5.2))
    ax.invert_yaxis()
    ax.set_xlim(-2, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=fs(5))
    ax.set_xlabel("share of replies with the flag (Wilson 95% CI)", fontsize=fs(5.2), labelpad=2, x=0.42)
    ax.tick_params(axis="y", length=0)
    clean(ax, ("top", "right", "left"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MCOL[m], ms=3.5, label=m) for m in MODELS],
              frameon=False, fontsize=fs(5), loc="lower right", ncol=1, handlelength=1.0, borderaxespad=0.2)
    tag(ax, "d", "Binary flags by model", x=-0.3)

    save(fig, "fig_behaviour_attributes")


# =============================================================================== Level 2
CODE_FAM = [("safety / stance", ["SEN", "AUR"]), ("tentative / directive", ["TEN", "DIR"]),
            ("advice", ["FIX", "RECT"]), ("topic", ["TSH"]), ("questions", ["QOP", "QCL"]),
            ("matching", ["LMT", "MEN"]), ("accurate empathy", ["VAC", "NAC", "ASAC", "SAC"]),
            ("inaccurate empathy", ["VIN", "NIN", "ASIN", "SIN"]), ("", ["INC"])]
CODE_ORDER = [c for _, cs in CODE_FAM for c in cs]


def level2():
    cov = pd.read_csv(TAB_B / "span_coverage.csv", index_col=0).loc[CODE_ORDER, MODELS]
    emp = pd.read_csv(TAB_B / "empathy_precision.csv").set_index("model")
    acc, inacc = emp["accurate"], emp["attempted"] - emp["accurate"]
    prec = emp["precision"]
    phis = {m: pd.read_csv(TAB_B / f"span_cooccurrence_{m}.csv", index_col=0) for m in MODELS}

    fig = plt.figure(figsize=(FIG_W_IN, 5.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 0.95, 0.95], wspace=0.5,
                  left=0.2, right=0.985, top=0.86, bottom=0.11)

    # (a) coverage heatmap
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cov.values, cmap=ltc_seq, vmin=0, vmax=18, aspect="auto")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs(4.6),
                    color="white" if v > 9 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(MODELS, fontsize=fs(5.4), rotation=35, ha="left", rotation_mode="anchor")
    for t, m in zip(ax.get_xticklabels(), MODELS):
        t.set_color(MCOL[m])
    ax.set_yticks(range(len(CODE_ORDER))); ax.set_yticklabels(CODE_ORDER, fontsize=fs(5.2))
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    # family brackets, drawn in a column left of the code labels (x in axes fraction, y in data)
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    y0 = 0
    for fam, cs in CODE_FAM:
        y1 = y0 + len(cs)
        if fam:
            ax.plot([-0.3, -0.3], [y0 - 0.35, y1 - 0.65], color=MUTE, lw=0.8, clip_on=False, transform=tr)
            ax.text(-0.34, (y0 + y1 - 1) / 2, fam, ha="right", va="center", fontsize=fs(5), color=MUTE,
                    clip_on=False, transform=tr)
        if y1 < len(CODE_ORDER):
            ax.axhline(y1 - 0.5, color="white", lw=1.6)
        y0 = y1
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=fs(4.6), length=2)
    cb.set_label("% of reply words highlighted", fontsize=fs(5), labelpad=3)
    cb.outline.set_visible(False)
    figtag(fig, "a", "Span coverage, all 20 codes", 0.02, 0.95)

    # (b) empathy: accurate vs inaccurate instances, precision
    ax = fig.add_subplot(gs[1])
    y = np.arange(5)
    ax.barh(y - 0.19, [acc[m] for m in MODELS], height=0.36, color=GCOL["empathy"], label="accurate")
    ax.barh(y + 0.19, [inacc[m] for m in MODELS], height=0.36, color="#7fbfcb", label="inaccurate")
    for i, m in enumerate(MODELS):
        ax.text(max(acc[m], inacc[m]) + 40, i, f"precision {prec[m] * 100:.0f}%", va="center", ha="left",
                fontsize=fs(5.2), color=INK, fontweight="bold" if m == "Llama" else "normal")
    ax.set_yticks(y); ax.set_yticklabels(MODELS, fontsize=fs(5.4))
    for t, m in zip(ax.get_yticklabels(), MODELS):
        t.set_color(MCOL[m])
    ax.invert_yaxis()
    ax.set_xlim(0, 3000)
    ax.set_xticks([0, 1000, 2000])
    ax.set_xticklabels(["0", "1,000", "2,000"], fontsize=fs(5))
    ax.set_xlabel("empathy spans (instances)", fontsize=fs(5.4), labelpad=2)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=fs(5), loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.0)
    clean(ax, ("top", "right", "left"))
    tag(ax, "b", "Empathy precision", x=-0.3)

    # (c) co-occurrence: the strongest pairs, phi per model
    keys = ["TSH", "VIN", "DIR", "FIX", "RECT", "AUR", "TEN", "QOP", "QCL", "VAC", "NIN", "ASIN", "SIN", "LMT"]
    pairs = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs[(a, b)] = [phis[m].loc[a, b] for m in MODELS]
    top = sorted(pairs.items(), key=lambda kv: -np.mean(kv[1]))[:9]
    ax = fig.add_subplot(gs[2])
    for i, ((a, b), vals) in enumerate(top):
        ax.axhspan(i - 0.5, i + 0.5, color=FAINT if i % 2 == 0 else "white", zorder=0, lw=0)
        ax.plot([min(vals), max(vals)], [i, i], color="#c8cdd2", lw=2.2, solid_capstyle="round", zorder=1)
        for m, v in zip(MODELS, vals):
            ax.plot(v, i, "o", color=MCOL[m], ms=4, mec="white", mew=0.5, zorder=2)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{a} + {b}" for (a, b), _ in top], fontsize=fs(5.4))
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(["0", ".2", ".4", ".6"], fontsize=fs(5))
    ax.set_xlabel("φ in the same reply", fontsize=fs(5.4), labelpad=2)
    ax.tick_params(axis="y", length=0)
    clean(ax, ("top", "right", "left"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MCOL[m], ms=3.5, label=m) for m in MODELS],
              frameon=False, fontsize=fs(5), loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=0.8,
              columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    tag(ax, "c", "Codes that share a reply", x=-0.32)

    save(fig, "fig_behaviour_spans")


# =============================================================================== Level 4
BINS = [("1-2", 1, 2), ("3-4", 3, 4), ("5-6", 5, 6), ("7-8", 7, 8), ("9+", 9, 99)]


def load_multiturn():
    frames = []
    for corpus in ("carebench", "hope"):
        df = bm.load_annotated(corpus)
        for k, m in enumerate(MODELS, start=1):
            sub = pd.DataFrame({"corpus": corpus, "turn": pd.to_numeric(df["Turn"], errors="coerce"), "model": m})
            for a in ATTRS:
                sub[a] = pd.to_numeric(df[f"Response {k}_{a}_score"], errors="coerce")
            sub["words"] = df[f"{m} Output"].astype(str).str.split().str.len()
            frames.append(sub)
    d = pd.concat(frames, ignore_index=True).dropna(subset=["turn"])
    d["bin"] = pd.cut(d["turn"], bins=[0, 2, 4, 6, 8, 99], labels=[b for b, _, _ in BINS])
    d["FIXb"] = (d["FIX"] > 0).astype(float)
    return d


def level4():
    d = load_multiturn()
    qpct = pd.read_csv(TAB_B / "pct_reply_with_question_by_turn.csv").set_index("model").rename(index={"Human": "Human therapist"})
    auto = pd.read_csv(TAB_B / "advice_autocorrelation.csv").set_index("model").rename(
        columns={"null_mean": "shuffle_mean", "p_vs_null": "p_vs_shuffle"})
    labels = [b for b, _, _ in BINS]
    xs = np.arange(5)

    fig = plt.figure(figsize=(FIG_W_IN, 5.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.62, wspace=0.42,
                  left=0.07, right=0.985, top=0.865, bottom=0.1)

    # (a) four attribute trends
    panels = [("QOC", "R7: QOC"), ("TD", "R3: TD"), ("TN", "R6: TN"), ("FIXb", "R4: FIX (>0)")]
    axes = []
    for i, (a, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2] if i < 4 else None)
        axes.append(ax)
        for m in MODELS:
            g = d[d.model == m].groupby("bin", observed=False)[a].mean()
            ax.plot(xs, g.loc[labels].values, color=MCOL[m], lw=1.3, marker="o", ms=2.8, mec="white", mew=0.4)
        pooled = d.groupby("bin", observed=False)[a].mean().loc[labels].values
        ax.plot(xs, pooled, color=INK, lw=1.1, ls="--", zorder=0)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=fs(5))
        ax.set_title(title, fontsize=fs(5.6), color=INK, loc="left", pad=3)
        ax.tick_params(axis="y", labelsize=fs(5))
        if i >= 2:
            ax.set_xlabel("turn", fontsize=fs(5.4), labelpad=2)
        clean(ax)
    figtag(fig, "a", "Mean clinician score by turn bin (dashed: pooled over models)", 0.03, 0.91)
    fig.legend(handles=[Line2D([], [], color=MCOL[m], lw=1.3, label=m) for m in MODELS],
               frameon=False, fontsize=fs(5.4), loc="lower right", bbox_to_anchor=(0.655, 0.945), ncol=5,
               handlelength=1.4, columnspacing=1.0)

    # (b) % replies with at least one question, vs therapist
    ax = fig.add_subplot(gs[0, 2])
    for m in MODELS:
        ax.plot(xs, qpct.loc[m, labels].values, color=MCOL[m], lw=1.3, marker="o", ms=2.8, mec="white", mew=0.4)
    ax.plot(xs, qpct.loc["Human therapist", labels].values, color=INK, lw=1.4, ls="--", marker="s", ms=2.8)
    ax.text(1.5, qpct.loc["Human therapist", "3-4"] + 6, "human therapist", fontsize=fs(5), color=INK, ha="center", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=fs(5))
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=fs(5))
    ax.set_xlabel("turn", fontsize=fs(5.4), labelpad=2)
    clean(ax)
    tag(ax, "b", "Replies with ≥1 question", x=-0.2)

    # (c) advice persistence: lag-1 autocorrelation vs shuffled-turn null
    ax = fig.add_subplot(gs[1, 2])
    for i, m in enumerate(MODELS):
        r = auto.loc[m]
        ax.plot([r.shuffle_mean, r.lag1_autocorr], [i, i], color="#c8cdd2", lw=2.2, solid_capstyle="round", zorder=1)
        ax.plot(r.shuffle_mean, i, "o", color="white", mec=MCOL[m], mew=1.1, ms=4.2, zorder=2)
        ax.plot(r.lag1_autocorr, i, "o", color=MCOL[m], ms=4.6, mec="white", mew=0.4, zorder=3)
        ax.text(r.lag1_autocorr + 0.012, i, f"p = {r.p_vs_shuffle:.3f}" if r.p_vs_shuffle >= 0.01 else "p < .01",
                fontsize=fs(4.8), color=MUTE, va="center")
    ax.set_yticks(range(5)); ax.set_yticklabels(MODELS, fontsize=fs(5.4))
    for t, m in zip(ax.get_yticklabels(), MODELS):
        t.set_color(MCOL[m])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.95)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_xticklabels([".5", ".6", ".7", ".8", ".9"], fontsize=fs(5))
    ax.set_xlabel("lag-1 autocorrelation of advice density", fontsize=fs(5.4), labelpad=2)
    ax.tick_params(axis="y", length=0)
    clean(ax, ("top", "right", "left"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=4, label="observed"),
                       Line2D([], [], marker="o", ls="", color="white", mec=INK, mew=1.0, ms=4, label="shuffled turns")],
              frameon=False, fontsize=fs(5), loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.0,
              columnspacing=1.0, borderaxespad=0.0)
    tag(ax, "c", "Advice carries over turns", x=-0.2)

    save(fig, "fig_behaviour_turns")


# =============================================================================== Level 5
SPEAKERS = MODELS + ["Human therapist"]
SPLAB = {**{m: m for m in MODELS}, "Human therapist": "Therapist"}


def word_counts_in_spans():
    """Word counts per model over the unique highlighted spans (tables/behaviour/span_word_counts.csv)."""
    import collections
    W = pd.read_csv(TAB_B / "span_word_counts.csv", index_col=0)
    return {m: collections.Counter(W[m].to_dict()) for m in MODELS}


def watermark_words(cnt, k=3, min_count=30, alpha=0.5):
    """Top-k words per model by log-odds against the other four models."""
    N = {m: sum(c.values()) for m, c in cnt.items()}
    vocab = set().union(*[set(c) for c in cnt.values()])
    out = {}
    for m in MODELS:
        rows = []
        for w in vocab:
            c = cnt[m][w]
            if c < min_count:
                continue
            o = sum(cnt[x][w] for x in MODELS if x != m)
            No = sum(N[x] for x in MODELS if x != m)
            lo = np.log((c + alpha) / (N[m] - c + alpha)) - np.log((o + alpha) / (No - o + alpha))
            rows.append((lo, w))
        rows.sort(reverse=True)
        out[m] = [w for _, w in rows[:k]]
    return out, N


def level5():
    """How the five models talk: shared phrasebook, watermark words, template load, copy-paste sentences."""
    share = pd.read_csv(TAB_B / "template_sharing.csv")
    conc = pd.read_csv(TAB_B / "template_concentration.csv").set_index("model")
    reuse = pd.read_csv(TAB_B / "fourgram_reuse.csv", index_col=0).rename(index={"Human": "Human therapist"})
    paste = pd.read_csv(TAB_B / "copy_paste_spans.csv")
    cnt = word_counts_in_spans()
    wm, Ntok = watermark_words(cnt)

    fig = plt.figure(figsize=(FIG_W_IN, 6.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.9, 1.1], height_ratios=[1.0, 0.8], hspace=0.5, wspace=0.42,
                  left=0.1, right=0.985, top=0.93, bottom=0.085)

    # ---- (a) Jaccard overlap of top-50 four-word phrases
    ax = fig.add_subplot(gs[0, 0])
    J = pd.DataFrame(np.nan, index=MODELS, columns=MODELS)
    N = pd.DataFrame(0, index=MODELS, columns=MODELS)
    for _, r in share.iterrows():
        J.loc[r.model_a, r.model_b] = J.loc[r.model_b, r.model_a] = r.jaccard_top50
        N.loc[r.model_a, r.model_b] = N.loc[r.model_b, r.model_a] = int(r.n_shared)
    ax.imshow(np.ma.masked_invalid(J.values.astype(float)), cmap=ltc_seq, vmin=0, vmax=0.4, aspect="auto")
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=FAINT, lw=0))
                continue
            v = J.values[i, j]
            col = "white" if v > 0.2 else INK
            ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=fs(6.2), color=col,
                    fontweight="bold" if v > 0.2 else "normal")
            ax.text(j, i + 0.24, f"{N.values[i, j]} shared", ha="center", va="center", fontsize=fs(4.8),
                    color=col if v > 0.2 else MUTE)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(MODELS, fontsize=fs(6)); ax.set_yticklabels(MODELS, fontsize=fs(6))
    for t, m in zip(ax.get_xticklabels(), MODELS):
        t.set_color(MCOL[m])
    for t, m in zip(ax.get_yticklabels(), MODELS):
        t.set_color(MCOL[m])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    qg = share[(share.model_a == "Qwen") & (share.model_b == "Gemini")].iloc[0]
    shared = [x.strip() for x in qg.shared_examples.split("|")]
    ax.text(0.0, -0.04, "Qwen–Gemini shared phrases include\n" + " · ".join(f"“{x}”" for x in shared[:2])
            + "\n" + " · ".join(f"“{x}”" for x in shared[2:]),
            transform=ax.transAxes, fontsize=fs(5.2), color=MUTE, ha="left", va="top", linespacing=1.35)
    tag(ax, "a", x=-0.22, pad=16)

    # ---- (b) watermark words: who says it? volume-adjusted share of use, one stacked bar per word
    ax = fig.add_subplot(gs[0, 1])
    rows = [(m, w) for m in MODELS for w in wm[m]]
    yy, y = [], 0.0
    for i, (m, w) in enumerate(rows):
        if i and rows[i - 1][0] != m:
            y += 0.55                                     # gap between owner groups
        yy.append(y)
        y += 1.0
    for (m, w), yv in zip(rows, yy):
        rate = np.array([cnt[x][w] / Ntok[x] for x in MODELS])
        sh = rate / rate.sum() * 100
        left = 0.0
        for x, v in zip(MODELS, sh):
            ax.barh(yv, v, left=left, height=0.78, color=MCOL[x], edgecolor="white", linewidth=0.6,
                    alpha=1.0 if x == m else 0.45)
            left += v
        own = sh[MODELS.index(m)]
        ax.text(sh[MODELS.index(m)] / 2 if MODELS.index(m) == 0 else
                sum(sh[:MODELS.index(m)]) + own / 2, yv, f"{own:.0f}%", ha="center", va="center",
                fontsize=fs(5.4), color="white", fontweight="bold")
        ax.text(101.5, yv, f"×{cnt[m][w]}", ha="left", va="center", fontsize=fs(5.2), color=MUTE)
    ax.set_yticks(yy)
    ax.set_yticklabels([w for _, w in rows], fontsize=fs(6.2), style="italic")
    for t, (m, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(MCOL[m])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=fs(5.4))
    ax.set_xlabel("share of use across the five models (span-volume adjusted)", fontsize=fs(5.4), labelpad=3)
    ax.tick_params(axis="y", length=0)
    clean(ax, ("top", "right", "left"))
    ax.legend(handles=[Patch(color=MCOL[m], label=m) for m in MODELS], frameon=False, fontsize=fs(5.4), ncol=5,
              loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=0.9, handletextpad=0.4,
              borderaxespad=0.0)
    tag(ax, "b", x=-0.2, pad=16)

    # ---- (c) template load: 4-gram self-reuse (multi-turn) and top-20 phrase coverage
    gsc = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.3, width_ratios=[1, 0.85])
    yv = np.arange(len(SPEAKERS))
    ax = fig.add_subplot(gsc[0])
    mt = reuse[["carebench", "hope"]].mean(axis=1)
    cols = [MCOL.get(sp, INK) for sp in SPEAKERS]
    ax.barh(yv, [mt[sp] for sp in SPEAKERS], color=cols, height=0.66)
    for i, sp in enumerate(SPEAKERS):
        ax.text(mt[sp] + 0.6, i, f"{mt[sp]:.0f}%", va="center", fontsize=fs(5.6), color=INK)
    ax.set_yticks(yv); ax.set_yticklabels([SPLAB[sp] for sp in SPEAKERS], fontsize=fs(6))
    for t, c in zip(ax.get_yticklabels(), cols):
        t.set_color(c)
    ax.invert_yaxis()
    ax.set_xlim(0, 36); ax.set_xticks([0, 10, 20, 30]); ax.set_xticklabels(["0", "10", "20", "30%"], fontsize=fs(5.4))
    ax.set_xlabel("own four-word phrases\nreused across replies", fontsize=fs(5.6), labelpad=3)
    ax.tick_params(axis="y", length=0)
    clean(ax, ("top", "right", "left"))
    tag(ax, "c", x=-0.45, pad=6)

    ax = fig.add_subplot(gsc[1], sharey=ax)
    cv = conc["top20_phrase_coverage"] * 100
    ax.barh(yv[:5], [cv[m] for m in MODELS], color=[MCOL[m] for m in MODELS], height=0.66)
    for i, m in enumerate(MODELS):
        ax.text(cv[m] + 0.3, i, f"{cv[m]:.1f}%", va="center", fontsize=fs(5.6), color=INK)
    ax.text(0.3, 5, "n/a (no span codes)", va="center", fontsize=fs(5.2), color=MUTE)
    ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15]); ax.set_xticklabels(["0", "5", "10", "15%"], fontsize=fs(5.4))
    ax.set_xlabel("spans built on the model's\nown top-20 phrases", fontsize=fs(5.6), labelpad=3)
    ax.tick_params(axis="y", length=0, labelleft=False)
    clean(ax, ("top", "right", "left"))

    # ---- (d) copy-paste sentences: one dot per user who received the identical highlighted sentence
    ax = fig.add_subplot(gs[1, 1])
    skip = paste.text.str.contains(r"988|741741|1-800|country", regex=True)
    rows = paste[~skip].sort_values("n_prompts", ascending=False)
    rows = rows[rows.text.str.len() <= 56].head(7).reset_index(drop=True)
    ax.set_xlim(0, 1); ax.set_ylim(len(rows) - 0.3, -0.6)
    ax.axis("off")
    x0, step = 0.145, 0.03
    for i, r in rows.iterrows():
        ax.text(0.0, i - 0.2, r.model, ha="left", va="center", fontsize=fs(5.6), color=MCOL[r.model], fontweight="bold")
        ax.text(x0, i - 0.2, f"“{r.text.strip()}”", ha="left", va="center", fontsize=fs(4.9), color=INK)
        n = int(r.n_prompts)
        ax.scatter([x0 + k * step for k in range(n)], [i + 0.22] * n, s=fs(4.4) ** 2, color=MCOL[r.model],
                   edgecolor="white", linewidth=0.5, zorder=3, clip_on=False)
        ax.text(x0 + n * step + 0.005, i + 0.22, f"{n} users", ha="left", va="center", fontsize=fs(5.2), color=MUTE)
    ax.text(0.0, len(rows) - 0.35, "● = one user who received this exact highlighted sentence", ha="left", va="top",
            fontsize=fs(5.2), color=MUTE)
    tag(ax, "d", x=-0.02, pad=6)

    save(fig, "fig_behaviour_language")


if __name__ == "__main__":
    level1()
    level2()
    level4()
    level5()
