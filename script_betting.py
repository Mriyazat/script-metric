#!/usr/bin/env python3
"""
SCRIPT-Seq — anytime-valid scriptedness testing by betting.

    SCRIPT (script_metric.py) answers: given this corpus, is the arrangement
    structure real, and how big is it?

    SCRIPT-Seq answers: watching annotated replies arrive one at a time, when
    may I stop and declare — with a false-positive rate that holds at every
    single moment I look?

Those are different questions, and the second is the one every deployment use
of the metric actually asks: a scriptedness ceiling to audit against, a
regression test across a model release, a monitor on a live system. A
fixed-sample z answers the first and is invalid for the second: recomputing z
as data accumulates and stopping the first time it clears 1.96 has a
false-positive rate that grows with the number of looks, not one bounded by
0.05.

METHOD
------
Testing by betting (Shafer 2021; Ramdas et al. 2023), in the form Chen & Wang
(arXiv:2410.22318) use for online LLM-text detection, over a no-regret bettor
(Online Newton Step) in the coin-betting reduction of Orabona & Pal (2016) and
Orabona & Tommasi (NeurIPS 2017).

A gambler starts with wealth 1 and bets on a payoff that is mean-zero under the
null. If the wealth stays a non-negative supermartingale, Ville's inequality
gives, simultaneously for all t,

    P( exists t : W_t >= 1/alpha )  <=  alpha            under H0

so "reject the first time W_t >= 1/alpha" is a level-alpha test that may be
peeked at continuously, stopped early, or run forever.

WHY IT FITS SCRIPT EXACTLY
--------------------------
The construction needs a payoff with mean exactly zero under the null, and no
distributional assumption. SCRIPT's null already *is* an exchangeability
statement: within a reply, the labels are exchangeable across that reply's own
positions. So at round t, with the profile built only from replies 1..t-1
(hence measurable w.r.t. the past):

    e_t = loglik(reply t)  -  mean over K within-reply label shuffles of it

Under H0 the real reply is itself distributed as one of those shuffles, so
E[e_t | F_{t-1}] = 0 exactly. Nothing is assumed about the label distribution,
the reply lengths, the annotator, or the sample size. Under H1 the real
arrangement matches the accumulated profile better than a shuffle does, so
E[e_t] = delta > 0 and the wealth compounds.

The edge delta is in nats per event, not on the SCRIPT scale. It is monotone in
scriptedness but is not the same number; `calibrate` fits the mapping so a
SCRIPT ceiling can be converted into an edge threshold.

FOUR TESTS
----------
    test_structure   H0: delta <= 0      "is there any arrangement structure?"
    test_threshold   H0: delta <= eps    "does it exceed a ceiling?"      (composite)
    test_two_sample  H0: delta_A <= delta_B   "is A more scripted than B?" (paired)
    confidence_seq   an anytime-valid interval for delta, valid at all t

Usage
-----
    python script_betting.py test    spans.csv [--alpha 0.05] [--eps 0.0]
    python script_betting.py compare a.csv b.csv [--alpha 0.05]

`spans.csv` takes the same columns as script_metric.py: response_id, label,
and position (or start + response_length).

Dependencies: numpy, pandas. Single file.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from script_metric import load_spans

# ONS step size for the log-wealth loss; the standard 2/(2 - ln 3) constant
ONS_GAMMA = 2.0 / (2.0 - np.log(3.0))
DEFAULT_K = 20            # within-reply shuffles used to form the payoff
DEFAULT_BINS = 10
SMOOTH = 0.5              # add-alpha smoothing on the streaming profile


# ------------------------------------------------------- the running profile

class StreamingProfile:
    """Position and transition counts, updated one reply at a time.

    Only replies seen *before* the current round enter the tables, which is
    what makes the payoff's conditional mean exactly zero under the null.
    """

    def __init__(self, labels, n_bins=DEFAULT_BINS):
        self.labels = list(labels)
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.n_bins = n_bins
        self.pos = np.zeros((len(self.labels), n_bins))
        self.tr = np.zeros((len(self.labels), len(self.labels)))
        self._dirty = True

    def _refresh(self):
        P = self.pos + SMOOTH
        self._logP = np.log(P / P.sum(0, keepdims=True))
        T = self.tr + SMOOTH
        self._logT = np.log(T / T.sum(1, keepdims=True))
        self._dirty = False

    def loglik(self, lab, bins) -> float:
        """Mean per-event log P(label|bin) + log P(label|previous label)."""
        if self._dirty:
            self._refresh()
        total = float(self._logP[lab, bins].sum())
        if len(lab) > 1:
            total += float(self._logT[lab[:-1], lab[1:]].sum())
        return total / len(lab)

    def update(self, lab, bins) -> None:
        np.add.at(self.pos, (lab, bins), 1.0)
        if len(lab) > 1:
            np.add.at(self.tr, (lab[:-1], lab[1:]), 1.0)
        self._dirty = True


def edge(profile: StreamingProfile, lab, bins, rng, k=DEFAULT_K) -> float:
    """e_t = loglik(real) - mean_shuffle(loglik). Mean zero under H0.

    The shuffles permute the reply's own labels across its own positions, so
    the reply's behaviour mix, its length and its positions are all held fixed
    and only the label-to-position pairing varies — the same operation the
    published null performs, applied to a single reply.
    """
    real = profile.loglik(lab, bins)
    sh = np.empty(k)
    for i in range(k):
        sh[i] = profile.loglik(rng.permutation(lab), bins)
    return real - float(sh.mean())


# ------------------------------------------------------------- the bettor

class Bettor:
    """One-sided ONS bettor on a mean-zero-under-H0 payoff.

    Wealth W_t = W_{t-1} (1 + theta_t * e_t) with theta_t >= 0 constrained to
    [0, 1/(2 d_t)], where d_t bounds |e_{t+1}|. The constraint is what keeps
    the wealth non-negative, which is what Ville's inequality requires; the
    one-sidedness makes this a test of delta > 0 rather than delta != 0.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.theta = 0.0
        self.a = 1.0
        self.wealth = 1.0
        self.max_wealth = 1.0
        self.emax = 0.0
        self.t = 0
        self.tau = None
        self.trace = []

    def observe(self, e: float) -> float:
        self.t += 1
        self.wealth *= max(1.0 + self.theta * e, 0.0)
        self.max_wealth = max(self.max_wealth, self.wealth)
        self.trace.append(self.wealth)
        if self.tau is None and self.wealth >= 1.0 / self.alpha:
            self.tau = self.t

        # ONS on l(theta) = -ln(1 + theta e); gradient -e/(1 + theta e)
        denom = 1.0 + self.theta * e
        z = -e / denom if abs(denom) > 1e-12 else 0.0
        self.a += z * z
        self.emax = max(self.emax, abs(e))
        bound = 1.0 / (2.0 * max(2.0 * self.emax, 1e-6))
        self.theta = float(np.clip(self.theta - ONS_GAMMA * z / self.a,
                                   0.0, bound))
        return self.wealth

    @property
    def rejected(self) -> bool:
        return self.tau is not None


# ------------------------------------------------------------------- tests

def _prepare(df: pd.DataFrame, n_bins=DEFAULT_BINS, seed=0, shuffle_order=True):
    """-> (label index arrays, bin arrays) per reply, and the label alphabet."""
    d = df.sort_values(["response_id", "position"])
    labels = sorted(d.label.astype(str).unique())
    index = {l: i for i, l in enumerate(labels)}
    replies = []
    for _, g in d.groupby("response_id", sort=False):
        if len(g) < 2:
            continue
        lab = np.array([index[str(x)] for x in g.label])
        bins = np.minimum((g.position.to_numpy() * n_bins).astype(int),
                          n_bins - 1)
        replies.append((lab, bins))
    if shuffle_order:
        np.random.default_rng(seed).shuffle(replies)
    return replies, labels


def test_structure(df, alpha=0.05, eps=0.0, k=DEFAULT_K, n_bins=DEFAULT_BINS,
                   seed=0, max_t=None, shuffle_order=True):
    """H0: delta <= eps. Reject = 'arrangement structure exceeds eps'.

    eps = 0 is the plain existence test; eps > 0 is the deployment-threshold
    test, in which a system only trips the alarm if its edge clears the ceiling
    rather than merely being non-zero.
    """
    replies, labels = _prepare(df, n_bins, seed, shuffle_order)
    rng = np.random.default_rng(seed + 1)
    prof = StreamingProfile(labels, n_bins)
    bettor = Bettor(alpha)
    edges = []
    for lab, bins in (replies[:max_t] if max_t else replies):
        e = edge(prof, lab, bins, rng, k)
        edges.append(e)
        bettor.observe(e - eps)
        prof.update(lab, bins)
    edges = np.array(edges)
    return dict(rejected=bettor.rejected, tau=bettor.tau,
                n_replies=len(edges), wealth=bettor.wealth,
                max_wealth=bettor.max_wealth,
                edge_mean=float(edges.mean()) if len(edges) else np.nan,
                edge_se=float(edges.std(ddof=1) / np.sqrt(len(edges)))
                if len(edges) > 1 else np.nan,
                alpha=alpha, eps=eps, trace=bettor.trace)


def test_two_sample(df_a, df_b, alpha=0.05, k=DEFAULT_K, n_bins=DEFAULT_BINS,
                    seed=0, max_t=None):
    """H0: delta_A <= delta_B. Reject = 'A is more scripted than B'.

    Each system keeps its own streaming profile *and its own label alphabet*,
    so this compares how well each system's own accumulated routine predicts
    its own next reply — not whether one system's replies look like the
    other's. Each edge is already normalised against that system's own
    within-reply shuffles, so the paired difference is mean-zero under
    equality, which is all the betting construction needs.
    """
    reps_a, labs_a = _prepare(df_a, n_bins, seed)
    reps_b, labs_b = _prepare(df_b, n_bins, seed)
    prof_a = StreamingProfile(labs_a, n_bins)
    prof_b = StreamingProfile(labs_b, n_bins)
    rng = np.random.default_rng(seed + 1)
    bettor = Bettor(alpha)
    n = min(len(reps_a), len(reps_b))
    if max_t:
        n = min(n, max_t)
    diffs = []
    for i in range(n):
        la, ba = reps_a[i]
        lb, bb = reps_b[i]
        ea = edge(prof_a, la, ba, rng, k)
        eb = edge(prof_b, lb, bb, rng, k)
        diffs.append(ea - eb)
        bettor.observe(ea - eb)
        prof_a.update(la, ba)
        prof_b.update(lb, bb)
    diffs = np.array(diffs)
    return dict(rejected=bettor.rejected, tau=bettor.tau, n_pairs=n,
                wealth=bettor.wealth,
                diff_mean=float(diffs.mean()) if len(diffs) else np.nan,
                alpha=alpha, trace=bettor.trace)


def confidence_seq(df, alpha=0.05, grid=None, k=DEFAULT_K,
                   n_bins=DEFAULT_BINS, seed=0, max_t=None):
    """Anytime-valid LOWER confidence sequence for the edge delta.

    Obtained by inverting the threshold test: keep every eps whose one-sided
    test has not rejected, and report the smallest surviving value as a running
    lower bound on delta. Because the underlying tests are one-sided (H0:
    delta <= eps), this is a lower bound, not a two-sided interval — which is
    the useful direction here: "scriptedness is at least this much, and that
    statement holds at every t simultaneously with probability >= 1 - alpha".

    Unlike a fixed-sample confidence interval it may be watched continuously,
    stopped on, and reported at whatever moment is convenient, with no
    correction for the number of looks.

    Returns `lower` (the bound) and `upper` (the top of the search grid, which
    is a property of `grid`, not of the data — do not report it as a bound).
    """
    if grid is None:
        grid = np.linspace(-0.05, 1.0, 64)
    replies, labels = _prepare(df, n_bins, seed)
    if max_t:
        replies = replies[:max_t]
    rng = np.random.default_rng(seed + 1)
    prof = StreamingProfile(labels, n_bins)
    bettors = {float(g): Bettor(alpha) for g in grid}
    lows, highs = [], []
    for lab, bins in replies:
        e = edge(prof, lab, bins, rng, k)
        alive = []
        for g, b in bettors.items():
            if not b.rejected:
                b.observe(e - g)
            if not b.rejected:
                alive.append(g)
        prof.update(lab, bins)
        lows.append(min(alive) if alive else np.nan)
        highs.append(max(alive) if alive else np.nan)
    return dict(t=np.arange(1, len(lows) + 1), lower=np.array(lows),
                grid_top=np.array(highs), alpha=alpha)


def calibrate(edge_values, script_values):
    """Least-squares map from the betting edge (nats/event) to SCRIPT.

    The two are different quantities — SCRIPT is a corpus-level normalised
    mutual information, the edge is a per-reply predictive gain — so a
    deployment ceiling expressed in SCRIPT has to be converted before it can be
    used as `eps`. Returns slope, intercept, r, and `to_eps`.
    """
    x = np.asarray(edge_values, float)
    y = np.asarray(script_values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(slope=float(slope), intercept=float(intercept), r=r, n=len(x),
                to_eps=lambda s: (s - intercept) / slope)


# ------------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test", help="sequentially test one system")
    t.add_argument("spans")
    t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--eps", type=float, default=0.0,
                   help="edge ceiling; 0 tests for any structure at all")
    t.add_argument("--shuffles", type=int, default=DEFAULT_K)
    t.add_argument("--bins", type=int, default=DEFAULT_BINS)
    t.add_argument("--ci", action="store_true",
                   help="also report the anytime-valid confidence sequence")

    c = sub.add_parser("compare", help="sequentially test A more scripted than B")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--shuffles", type=int, default=DEFAULT_K)
    c.add_argument("--bins", type=int, default=DEFAULT_BINS)

    args = ap.parse_args()

    if args.cmd == "test":
        res = test_structure(load_spans(args.spans), alpha=args.alpha,
                             eps=args.eps, k=args.shuffles, n_bins=args.bins)
        out = {k: v for k, v in res.items() if k != "trace"}
        out["verdict"] = ("templated: edge exceeds eps"
                          if res["rejected"] else "not established")
        if args.ci:
            cs = confidence_seq(load_spans(args.spans), alpha=args.alpha,
                                k=args.shuffles, n_bins=args.bins)
            out["lower_confidence_bound_on_delta"] = round(
                float(cs["lower"][-1]), 4)
        print(json.dumps(out, indent=2, default=float))

    elif args.cmd == "compare":
        res = test_two_sample(load_spans(args.a), load_spans(args.b),
                              alpha=args.alpha, k=args.shuffles,
                              n_bins=args.bins)
        out = {k: v for k, v in res.items() if k != "trace"}
        out["verdict"] = ("A is more scripted than B" if res["rejected"]
                          else "no difference established")
        print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
