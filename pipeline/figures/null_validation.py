#!/usr/bin/env python3
"""Known-answer null validation: composition and small-sample sweeps on content-driven generators."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.common.paths import FIG, TAB

from scriptmetric import metric as sm

N_BINS = 10
N_SHUFFLES = 200
N_DRAWS = 12                     # repetitions per sweep point

# The two panels are independent demonstrations, so each picks the synthetic
# corpus that makes its own artefact legible. Only the shape of the curves
# carries meaning: the absolute height of raw R is a property of the label
# count and event density, not of the metric.
COMP = dict(n_labels=4, events_per_response=6, n_resp=160)
SIZE = dict(n_labels=3, events_per_response=10)

RAW_COL, SCRIPT_COL = "#c0392b", "#2c7bb6"


def synth(n_resp: int, dominant: float, n_labels: int,
          events_per_response: int, rng) -> pd.DataFrame:
    """A corpus with true R = 0 and a controlled label mix.

    `dominant` is the probability mass on label 0; the rest is spread evenly.
    Positions are uniform and drawn independently of the label, and the label
    is drawn independently of the previous label, so neither the choreography
    nor the momentum term has any real signal to find.
    """
    rest = (1.0 - dominant) / (n_labels - 1)
    p = np.array([dominant] + [rest] * (n_labels - 1))
    n = n_resp * events_per_response
    return pd.DataFrame({
        "response_id": np.repeat(np.arange(n_resp), events_per_response),
        "label": rng.choice(n_labels, size=n, p=p).astype(str),
        "position": rng.uniform(0, 1, size=n),
    }).sort_values(["response_id", "position"]).reset_index(drop=True)


def sweep(points, make, label) -> pd.DataFrame:
    rows = []
    for x in points:
        raws, scripts = [], []
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(hash((label, float(x), draw)) % 2**32)
            res, _ = sm.compute(make(x, rng), n_bins=N_BINS,
                                n_shuffles=N_SHUFFLES, seed=draw)
            raws.append(res["R_raw"])
            scripts.append(res["SCRIPT"])
        rows.append(dict(sweep=label, x=x,
                         raw_mean=np.mean(raws), raw_sd=np.std(raws),
                         script_mean=np.mean(scripts), script_sd=np.std(scripts)))
        print(f"  {label} x={x:<7} raw R={np.mean(raws):.4f}  "
              f"SCRIPT={np.mean(scripts):+.4f}", flush=True)
    return pd.DataFrame(rows)


def band(ax, x, mean, sd, color, label, logx=False):
    ax.plot(x, mean, "-o", ms=4, lw=1.8, color=color, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)
    if logx:
        ax.set_xscale("log")


def main() -> None:
    skews = [round(v, 3) for v in
             np.linspace(1.0 / COMP["n_labels"], 0.90, 9)]
    sizes = [10, 20, 40, 80, 160, 320]

    print("== (a) composition sweep, N fixed ==", flush=True)
    A = sweep(skews, lambda d, rng: synth(
        COMP["n_resp"], d, COMP["n_labels"],
        COMP["events_per_response"], rng), "composition")
    print("== (b) small-sample sweep, uniform labels ==", flush=True)
    B = sweep(sizes, lambda n, rng: synth(
        int(n), 1.0 / SIZE["n_labels"], SIZE["n_labels"],
        SIZE["events_per_response"], rng), "sample size")
    pd.concat([A, B]).to_csv(TAB / "null_validation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    ax = axes[0]
    band(ax, A.x, A.raw_mean, A.raw_sd, RAW_COL, "raw $R$ (uncalibrated)")
    band(ax, A.x, A.script_mean, A.script_sd, SCRIPT_COL,
         r"SCRIPT $= R - \overline{R^{(s)}}$")
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.set_xlabel("dominant-label fraction  (composition skew)")
    ax.set_ylabel("structural share")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    band(ax, B.x, B.raw_mean, B.raw_sd, RAW_COL, "raw $R$ (uncalibrated)",
         logx=True)
    band(ax, B.x, B.script_mean, B.script_sd, SCRIPT_COL,
         r"SCRIPT $= R - \overline{R^{(s)}}$", logx=True)
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.set_xlabel("number of responses $N$  (log scale)")
    ax.set_ylabel("structural share")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG / "fig_null_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_null_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    # The right test is not "is SCRIPT exactly 0" but "is it inside its own
    # sampling spread", since each point is a mean over N_DRAWS draws.
    both = pd.concat([A, B])
    off = (both.script_mean.abs() / (both.script_sd / np.sqrt(N_DRAWS))).max()
    print(f"\nlargest |SCRIPT| on a true-zero corpus: "
          f"{both.script_mean.abs().max():.4f}  "
          f"(worst deviation {off:.1f} SE from zero)")
    print(f"raw R peaks at {both.raw_mean.max():.4f}")
    if off > 4:
        print("WARNING: SCRIPT is drifting off zero by more than sampling "
              "noise — the null is not cancelling the artefacts as claimed.")
    print(f"saved {FIG / 'fig_null_validation.png'}")


if __name__ == "__main__":
    main()
