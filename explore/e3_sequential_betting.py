#!/usr/bin/env python3
"""Anytime-valid SCRIPT by betting (exploratory — not in the paper).

WHY. As published, SCRIPT is a fixed-sample statistic: collect the corpus,
permute, report z. That is fine for a corpus you already have, and wrong for
the use the paper actually proposes — auditing a deployed system as annotations
arrive. If you recompute z after every new reply and stop the first time it
crosses 1.96, your false-positive rate is not 5%; it climbs towards 1 with the
number of looks. This is exactly the failure Chen & Wang (2024) identify for
naive online adaptation of offline detectors.

THE FIX. Chen & Wang, "Online Detection of LLM-Generated Texts via Sequential
Hypothesis Testing by Betting" (arXiv:2410.22318), turn detection into a
betting game: a gambler bets on a mean-zero payoff, the wealth is a
non-negative supermartingale under the null, and Ville's inequality makes
`reject when wealth >= 1/alpha` a level-alpha test *simultaneously at every
time step*. Betting fractions come from a no-regret learner (Online Newton
Step), which is what buys power; the parameter-free spirit is Orabona &
Tommasi's coin-betting reduction (NeurIPS 2017).

WHAT MAKES IT FIT SCRIPT. The construction needs a payoff that is exactly
mean-zero under the null. SCRIPT's null already *is* an exchangeability
statement — within a response, labels are exchangeable across that response's
own positions. So:

    at round t, build the profile from responses 1..t-1   (F_{t-1}-measurable)
    a_t = per-event log-lik of reply t under that profile
    b_t = the same, averaged over K within-reply label shuffles of reply t
    g_t = b_t - a_t

Under H0 the real reply and its shuffles are exchangeable given F_{t-1}, so
E[g_t | F_{t-1}] = 0 with no distributional assumption whatsoever. No reference
corpus, no held-out data, no threshold to tune.

Three experiments:

  A  type-I error under H0 (labels globally permuted), betting vs naive peeking
  B  power: how many annotated replies to certify a real model, per model
  C  the wealth trajectory, for the figure

    python e3_sequential_betting.py [--alpha 0.05] [--streams 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from paths import EVENTS, MODELS, OUT   # noqa: E402

import script_metric as sm   # noqa: E402

N_BINS = 10
K_SHUFFLES = 20          # shuffles per reply used to form the mean-zero payoff
SMOOTH = 0.5
OUTDIR = OUT / "explore"
OUTDIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------- streaming profile

class Profile:
    """Position and transition counts, updated one reply at a time."""

    def __init__(self, labels):
        self.labels = labels
        self.l2i = {l: i for i, l in enumerate(labels)}
        self.pos = np.zeros((len(labels), N_BINS))
        self.tr = np.zeros((len(labels), len(labels)))

    def loglik(self, lab, bins) -> float:
        """Mean per-event log P(label|bin) + log P(label|previous)."""
        P = self.pos + SMOOTH
        P = P / P.sum(0, keepdims=True)
        T = self.tr + SMOOTH
        T = T / T.sum(1, keepdims=True)
        total = float(np.log(P[lab, bins]).sum())
        if len(lab) > 1:
            total += float(np.log(T[lab[:-1], lab[1:]]).sum())
        return total / len(lab)

    def update(self, lab, bins) -> None:
        np.add.at(self.pos, (lab, bins), 1.0)
        if len(lab) > 1:
            np.add.at(self.tr, (lab[:-1], lab[1:]), 1.0)


def payoff(prof: Profile, lab, bins, rng, k=K_SHUFFLES):
    """g_t = E_shuffle[loglik] - loglik(real). Mean zero under H0."""
    real = prof.loglik(lab, bins)
    sh = np.empty(k)
    for i in range(k):
        sh[i] = prof.loglik(rng.permutation(lab), bins)
    return float(sh.mean() - real)


# ------------------------------------------------------------- the bettor

def run_stream(replies, alpha=0.05, seed=0, max_t=None):
    """Sequential test by betting. -> (rejected, stopping time, wealth trace).

    `replies` is a list of (labels, bins) arrays in arrival order.
    ONS on l_t(theta) = -ln(1 - g_t theta), decision space clipped to
    [-1/(2 d_t), 1/(2 d_t)] so the wealth can never go negative — which is what
    Ville's inequality requires.
    """
    rng = np.random.default_rng(seed)
    labels = sorted({l for lab, _ in replies for l in lab})
    prof = Profile(labels)
    l2i = prof.l2i

    theta, a_t, W = 0.0, 1.0, 1.0
    d = 0.5                       # running bound on |g|; refined as we go
    gmax = 0.0
    trace, tau = [], None
    for t, (lab_raw, bins) in enumerate(replies[:max_t], start=1):
        lab = np.array([l2i[x] for x in lab_raw])
        g = payoff(prof, lab, bins, rng)

        W = W * (1.0 - g * theta)
        trace.append(W)
        if tau is None and W >= 1.0 / alpha:
            tau = t                                   # declare: templated

        # ONS update, then widen the decision space using the observed scale
        z = g / (1.0 - g * theta) if abs(1.0 - g * theta) > 1e-12 else 0.0
        a_t += z * z
        gmax = max(gmax, abs(g))
        d = max(2.0 * gmax, 1e-3)                     # hint on |g_{t+1}|
        bound = 1.0 / (2.0 * d)
        theta = float(np.clip(theta - (2.0 / (2 - np.log(3))) * z / a_t,
                              -bound, bound))
        prof.update(lab, bins)
    return tau is not None, tau, np.array(trace)


def run_naive(replies, alpha=0.05, seed=0, every=20, z_crit=1.96,
              n_shuffles=30):
    """The tempting-but-invalid alternative: recompute z and peek.

    Identical stopping rule in spirit — 'stop as soon as it looks significant'
    — but with the paper's fixed-sample statistic. Included to measure what
    peeking costs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t, (lab, bins) in enumerate(replies, start=1):
        rows.append((np.full(len(lab), t), lab, bins))
        if t % every or t < 20:
            continue
        rid = np.concatenate([r[0] for r in rows])
        L = np.concatenate([r[1] for r in rows])
        X = np.concatenate([r[2] for r in rows])
        labs = sorted(set(L))
        li = np.array([labs.index(x) for x in L])
        _, _, R = sm._metrics(li, X, rid, len(labs), N_BINS)
        null = np.array([sm._metrics(sm._shuffle_within(li, rid, rng), X, rid,
                                     len(labs), N_BINS)[2]
                         for _ in range(n_shuffles)])
        sd = null.std()
        if sd > 0 and (R - null.mean()) / sd >= z_crit:
            return True, t
    return False, None


# --------------------------------------------------------------- data prep

def as_replies(E: pd.DataFrame, rng):
    """-> list of (label array, bin array), one per reply, in random order."""
    E = E.sort_values(["response_id", "position"])
    out = []
    for _, g in E.groupby("response_id", sort=False):
        if len(g) < 2:
            continue
        bins = np.minimum((g.position.to_numpy() * N_BINS).astype(int),
                          N_BINS - 1)
        out.append((g.label.to_numpy(), bins))
    rng.shuffle(out)
    return out


def null_version(replies, rng):
    """H0 data: break the label-position link, keep everything else.

    Labels are permuted *within each reply*, so every reply keeps its own
    behaviour mix, its length and its positions. This is the null SCRIPT is
    defined against, materialised as a dataset.
    """
    return [(rng.permutation(lab), bins) for lab, bins in replies]


# -------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--streams", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=400)
    ap.add_argument("--exp", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--naive-streams", type=int, default=60,
                    help="how many H0 streams to also run the naive test on")
    args = ap.parse_args()

    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    thresh = 1.0 / args.alpha
    rows = []

    if args.exp in ("A", "all"):
        print("=" * 72)
        print(f"A — type-I error under H0 (alpha = {args.alpha}, "
              f"{args.streams} independent streams)")
        print("=" * 72)
        print("H0 data: each reply's labels permuted within itself, so there is"
              "\nno arrangement structure left to find. A valid test must "
              "reject\nat most alpha of the time, no matter how long we watch."
              "\n")
        rej_bet = rej_naive = 0
        for s in range(args.streams):
            rng = np.random.default_rng(1000 + s)
            reps = null_version(as_replies(E[E.model == "Claude"], rng), rng)
            ok, _, _ = run_stream(reps, args.alpha, seed=s, max_t=args.max_t)
            rej_bet += ok
            if s < min(args.streams, args.naive_streams):   # naive is slow
                n_ok, _ = run_naive(reps[:args.max_t], args.alpha, seed=s)
                rej_naive += n_ok
        n_naive = max(min(args.streams, args.naive_streams), 1)
        print(f"  betting (Ville)        false positives "
              f"{rej_bet}/{args.streams} = {rej_bet/args.streams:.3f}"
              f"   (guarantee: <= {args.alpha})")
        print(f"  naive peeking at z>1.96 false positives "
              f"{rej_naive}/{n_naive} = {rej_naive/n_naive:.3f}"
              f"   (no guarantee)")
        rows.append(dict(experiment="type-I", method="betting",
                         rate=rej_bet / args.streams, n=args.streams))
        rows.append(dict(experiment="type-I", method="naive peeking",
                         rate=rej_naive / n_naive, n=n_naive))

    if args.exp in ("B", "all"):
        print("\n" + "=" * 72)
        print("B — power: replies needed to certify a real model as templated")
        print("=" * 72)
        print(f"Stop the first time wealth >= 1/alpha = {thresh:.0f}.\n")
        for m in MODELS:
            taus, hits = [], 0
            for s in range(30):
                rng = np.random.default_rng(2000 + s)
                reps = as_replies(E[E.model == m], rng)
                ok, tau, _ = run_stream(reps, args.alpha, seed=s,
                                        max_t=args.max_t)
                hits += ok
                if ok:
                    taus.append(tau)
            med = int(np.median(taus)) if taus else None
            q90 = int(np.percentile(taus, 90)) if taus else None
            print(f"  {m:8s} detected {hits}/30 streams   "
                  f"median {med} replies   90th pct {q90}")
            rows.append(dict(experiment="power", method=m,
                             rate=hits / 30, n=30, median_tau=med, p90_tau=q90))

    if args.exp in ("C", "all"):
        print("\n" + "=" * 72)
        print("C — wealth trajectories")
        print("=" * 72)
        tr = {}
        for m in MODELS:
            rng = np.random.default_rng(7)
            _, _, w = run_stream(as_replies(E[E.model == m], rng), args.alpha,
                                 seed=7, max_t=args.max_t)
            tr[m] = w
        rng = np.random.default_rng(7)
        reps = null_version(as_replies(E[E.model == "Claude"], rng), rng)
        _, _, w0 = run_stream(reps, args.alpha, seed=7, max_t=args.max_t)
        tr["H0 (shuffled)"] = w0
        L = min(len(v) for v in tr.values())
        pd.DataFrame({k: v[:L] for k, v in tr.items()}).to_csv(
            OUTDIR / "betting_wealth.csv", index=False)
        for k, v in tr.items():
            cross = np.argmax(v >= thresh) + 1 if (v >= thresh).any() else None
            print(f"  {k:16s} final wealth {v[-1]:.3g}"
                  f"   crossed 1/alpha at t={cross}")
        print(f"\n  wrote {OUTDIR / 'betting_wealth.csv'}")

    if rows:
        pd.DataFrame(rows).to_csv(OUTDIR / "betting_summary.csv", index=False)
        print(f"\nwrote {OUTDIR / 'betting_summary.csv'}")


if __name__ == "__main__":
    main()
