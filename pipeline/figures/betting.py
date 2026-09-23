#!/usr/bin/env python3
"""SCRIPT-Seq figure: validity, stopping times, transfer. Reads the betting_*.csv tables."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.common.paths import EVENTS, FIG, MODELS, TAB

from scriptmetric import betting as sb

ALPHA = 0.05
MAX_T = 400
MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "figure.facecolor": "white"})
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4),
                             gridspec_kw=dict(wspace=0.3))

    # ---- (a) wealth trajectories -------------------------------------------
    ax = axes[0]
    for m in MODELS:
        d = E[E.model == m][["response_id", "label", "position"]]
        r = sb.test_structure(d, alpha=ALPHA, seed=7, max_t=MAX_T)
        ax.plot(np.arange(1, len(r["trace"]) + 1), r["trace"],
                color=MCOLORS[m], lw=1.6, label=m)
    rng = np.random.default_rng(7)
    base = E[E.model == "Claude"][["response_id", "label", "position"]].copy()
    base["label"] = base.groupby("response_id").label.transform(
        lambda s: rng.permutation(s.to_numpy()))
    r0 = sb.test_structure(base, alpha=ALPHA, seed=7, max_t=MAX_T)
    ax.plot(np.arange(1, len(r0["trace"]) + 1), r0["trace"], color="#888888",
            lw=2.2, ls="--", label="null (labels shuffled)", zorder=5)
    ax.axhline(1 / ALPHA, color="k", lw=1, ls=":")
    ax.text(398, 30, r"reject: wealth $\geq 1/\alpha = 20$", fontsize=8.5,
            color="#333333", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 1e17)
    ax.set_xlabel("annotated replies observed")
    ax.set_ylabel("wealth (log scale)")
    ax.legend(fontsize=7.6, frameon=False, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) sensitivity from the dilution sweep ---------------------------
    ax = axes[1]
    S = pd.read_csv(TAB / "betting_scaling.csv")
    keep = S.dropna(subset=["median_tau"])
    ax.plot(keep.signal_kept * 100, keep.median_tau, "-o", color="#d1495b",
            lw=1.8, ms=5)
    for _, r in keep.iterrows():
        ax.annotate(f"{int(r.detected)}/{int(r.streams)}",
                    (r.signal_kept * 100, r.median_tau),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                    color="#666666")
    miss = S[S.median_tau.isna()]
    if len(miss):
        ax.scatter(miss.signal_kept * 100, [MAX_T] * len(miss), marker="x",
                   s=55, color="#888888")
        ax.annotate("not detected\nwithin 400 replies",
                    (miss.signal_kept.max() * 100, MAX_T),
                    textcoords="offset points", xytext=(10, -6), fontsize=8,
                    color="#666666")
    ax.set_xlabel("share of the real template kept  (%)")
    ax.set_ylabel("replies to declare (median)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

    # ---- (c) anytime-valid lower bound -------------------------------------
    ax = axes[2]
    for m in MODELS:
        d = E[E.model == m][["response_id", "label", "position"]]
        cs = sb.confidence_seq(d, alpha=ALPHA, max_t=300)
        ax.plot(cs["t"], cs["lower"], color=MCOLORS[m], lw=1.5, label=m)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("annotated replies observed")
    ax.set_ylabel("lower bound on the edge (nats/event)")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

    fig.savefig(FIG / "fig_betting.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_betting.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'fig_betting.png'}")


if __name__ == "__main__":
    main()
