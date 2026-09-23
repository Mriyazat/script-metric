#!/usr/bin/env python3
"""
SCRIPT-Seq — anytime-valid scriptedness testing by betting.

    SCRIPT (scriptmetric/metric.py) answers: given this corpus, is the arrangement
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
    python -m scriptmetric.betting test    spans.csv [--alpha 0.05] [--eps 0.0]
    python -m scriptmetric.betting compare a.csv b.csv [--alpha 0.05]

`spans.csv` takes the same columns as scriptmetric/metric.py: response_id, label,
and position (or start + response_length).

Dependencies: numpy, pandas. Single file.
"""
import argparse
import json

import numpy as np
import pandas as pd

from scriptmetric.metric import load_spans

# ONS step size for the log-wealth loss; the standard 2/(2 - ln 3) constant
ONS_GAMMA = 2.0 / (2.0 - np.log(3.0))
DEFAULT_K = 20            # within-reply shuffles used to form the payoff
DEFAULT_BINS = 10
SMOOTH = 0.5              # add-alpha smoothing on the streaming profile
# Transition convention shared with scriptmetric.metric.compute: "exclude" (default)
# treats co-located spans (identical start position) as one multi-label slot
# and forms transitions only between distinct positions; "order" is the legacy
# convention in which co-located spans are chained in stable-sort order.
DEFAULT_TIES = "exclude"


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

    def loglik(self, lab, bins, tmask) -> float:
        """Mean per-event log P(label|bin) + log P(label|previous label).

        `tmask[i]` says whether the adjacent pair (i, i+1) is a transition; it
        depends on positions only, so it is the same for the real reply and for
        every shuffle of its labels."""
        if self._dirty:
            self._refresh()
        total = float(self._logP[lab, bins].sum())
        if len(lab) > 1:
            total += float(self._logT[lab[:-1][tmask], lab[1:][tmask]].sum())
        return total / len(lab)

    def payoff_bound(self, lab, bins, tmask) -> float:
        """d_t: a bound on |e_t| that is known BEFORE the reply's labels are seen
        in order.

        The null of `edge` is exchangeability of the reply's labels over its own
        positions, so it is conditional on the reply's label multiset and its
        positions; both may therefore enter the bound. Over all assignments of
        that multiset to those bins, the mean per-event log-likelihood ranges at
        most over the summed per-bin spread of log P(label | bin) across the
        labels present, plus the transition-cell spread across those labels for
        each transition. The real reply and every shuffle lie in that range, so
        |real - mean(shuffles)| <= d_t. Because the profile holds only earlier
        replies, d_t is F_{t-1}-measurable and a stake theta_t <= 1/d_t makes
        1 + theta_t e_t >= 0 with certainty, which is what Ville's inequality
        needs. (The legacy bound used the running maximum of |e|, which is not
        known in advance; see Bettor.)"""
        if self._dirty:
            self._refresh()
        labs = np.unique(lab)
        P = self._logP[np.ix_(labs, bins)]
        spread = float((P.max(0) - P.min(0)).sum())
        if len(lab) > 1 and tmask.any():
            T = self._logT[np.ix_(labs, labs)]
            spread += float(tmask.sum()) * float(T.max() - T.min())
        return spread / len(lab)

    def update(self, lab, bins, tmask) -> None:
        np.add.at(self.pos, (lab, bins), 1.0)
        if len(lab) > 1:
            np.add.at(self.tr, (lab[:-1][tmask], lab[1:][tmask]), 1.0)
        self._dirty = True


def edge(profile: StreamingProfile, lab, bins, tmask, rng, k=DEFAULT_K) -> float:
    """e_t = loglik(real) - mean_shuffle(loglik). Mean zero under H0.

    The shuffles permute the reply's own labels across its own positions, so
    the reply's behaviour mix, its length, its positions and its transition
    mask are all held fixed and only the label-to-position pairing varies —
    the same operation the published null performs, applied to a single reply.
    """
    real = profile.loglik(lab, bins, tmask)
    sh = np.empty(k)
    for i in range(k):
        sh[i] = profile.loglik(rng.permutation(lab), bins, tmask)
    return real - float(sh.mean())


# ------------------------------------------------------------- the bettor

class Bettor:
    """One-sided ONS bettor on a mean-zero-under-H0 payoff.

    Wealth W_t = W_{t-1} (1 + theta_t e_t), theta_t >= 0. Ville's inequality
    needs (W_t) to be a non-negative supermartingale, i.e. 1 + theta_t e_t >= 0
    with certainty, so theta_t must be capped by a bound on |e_t| that is known
    before e_t is observed.

    bound="reply" (default): the cap is 1/d_t with d_t the predictable bound
        returned by StreamingProfile.payoff_bound for the reply about to be
        scored (passed to observe); no guard is needed and the guarantee is exact.
    bound="running" (legacy): the cap is 1/(4 max_{s<=t} |e_s|). The running
        maximum is not known in advance, so observe() floors the wealth factor
        at zero; the guarantee then holds only on the event that the floor is
        never reached (it never was in the paper's runs). Kept for reproduction.

    The one-sidedness makes this a test of delta > 0 rather than delta != 0.
    """

    def __init__(self, alpha=0.05, bound="reply"):
        if bound not in ("reply", "running"):
            raise ValueError("bound must be 'reply' or 'running'")
        self.alpha = alpha
        self.bound = bound
        self.theta = 0.0
        self.a = 1.0
        self.wealth = 1.0
        self.max_wealth = 1.0
        self.emax = 0.0
        self.t = 0
        self.tau = None
        self.trace = []
        self.n_floor = 0          # legacy mode only: how often the floor was hit

    def observe(self, e: float, d: float = None) -> float:
        """Bet on payoff e. In bound="reply" mode `d` is the predictable bound on
        |e| for this round (|e| <= d must hold); it is required."""
        self.t += 1
        factor = 1.0 + self.theta * e
        if self.bound == "reply":
            if d is None:
                raise ValueError("bound='reply' needs the predictable bound d for this round")
            if abs(e) > d + 1e-9:
                raise ValueError(f"payoff {e:.4f} exceeds its bound {d:.4f}")
            # theta <= 1/d and |e| <= d  =>  factor >= 0 with certainty
        else:
            if factor < 0:
                self.n_floor += 1
            factor = max(factor, 0.0)
        self.wealth *= factor
        self.max_wealth = max(self.max_wealth, self.wealth)
        self.trace.append(self.wealth)
        if self.tau is None and self.wealth >= 1.0 / self.alpha:
            self.tau = self.t

        # ONS on l(theta) = -ln(1 + theta e); gradient -e/(1 + theta e)
        denom = 1.0 + self.theta * e
        z = -e / denom if abs(denom) > 1e-12 else 0.0
        self.a += z * z
        if self.bound == "reply":
            # cap for the NEXT round is set when its d is known (see set_cap);
            # here we take the unconstrained ONS step and clip at >= 0
            self.theta = max(float(self.theta - ONS_GAMMA * z / self.a), 0.0)
        else:
            self.emax = max(self.emax, abs(e))
            cap = 1.0 / (2.0 * max(2.0 * self.emax, 1e-6))
            self.theta = float(np.clip(self.theta - ONS_GAMMA * z / self.a, 0.0, cap))
        return self.wealth

    def stake(self, d: float) -> float:
        """bound="reply": clip the current stake to the cap 1/d for a round whose
        predictable payoff bound is d, and return it. A round with d = 0 has
        e = 0 exactly, so any stake is safe and the stake is left unchanged."""
        if self.bound == "reply" and d > 0:
            self.theta = min(self.theta, 1.0 / d)
        return self.theta

    @property
    def rejected(self) -> bool:
        return self.tau is not None


# ------------------------------------------------------------------- tests

def _prepare(df: pd.DataFrame, n_bins=DEFAULT_BINS, seed=0, shuffle_order=True,
             ties=DEFAULT_TIES):
    """-> (label indices, bins, transition mask) per reply, and the alphabet.

    Replies with a single located span carry no arrangement information and
    are skipped. The transition mask follows `ties` (see DEFAULT_TIES)."""
    d = df.sort_values(["response_id", "position"], kind="stable")
    labels = sorted(d.label.astype(str).unique())
    index = {l: i for i, l in enumerate(labels)}
    replies = []
    for _, g in d.groupby("response_id", sort=False):
        if len(g) < 2:
            continue
        lab = np.array([index[str(x)] for x in g.label])
        pos = g.position.to_numpy()
        bins = np.minimum((pos * n_bins).astype(int), n_bins - 1)
        tmask = (pos[1:] != pos[:-1]) if ties == "exclude" \
            else np.ones(len(lab) - 1, dtype=bool)
        replies.append((lab, bins, tmask))
    if shuffle_order:
        np.random.default_rng(seed).shuffle(replies)
    return replies, labels


def test_structure(df, alpha=0.05, eps=0.0, k=DEFAULT_K, n_bins=DEFAULT_BINS,
                   seed=0, max_t=None, shuffle_order=True, ties=DEFAULT_TIES,
                   bound="reply"):
    """H0: delta <= eps. Reject = 'arrangement structure exceeds eps'.

    eps = 0 is the plain existence test; eps > 0 is the deployment-threshold
    test, in which a system only trips the alarm if its edge clears the ceiling
    rather than merely being non-zero.
    """
    replies, labels = _prepare(df, n_bins, seed, shuffle_order, ties)
    rng = np.random.default_rng(seed + 1)
    prof = StreamingProfile(labels, n_bins)
    bettor = Bettor(alpha, bound)
    edges = []
    for lab, bins, tmask in (replies[:max_t] if max_t else replies):
        d = prof.payoff_bound(lab, bins, tmask) + abs(eps)   # |e - eps| <= d_t + |eps|
        bettor.stake(d)
        e = edge(prof, lab, bins, tmask, rng, k)
        edges.append(e)
        bettor.observe(e - eps, d)
        prof.update(lab, bins, tmask)
    edges = np.array(edges)
    return dict(rejected=bettor.rejected, tau=bettor.tau,
                n_replies=len(edges), wealth=bettor.wealth,
                max_wealth=bettor.max_wealth,
                edge_mean=float(edges.mean()) if len(edges) else np.nan,
                edge_se=float(edges.std(ddof=1) / np.sqrt(len(edges)))
                if len(edges) > 1 else np.nan,
                alpha=alpha, eps=eps, trace=bettor.trace,
                bound=bound, n_floor=bettor.n_floor)


def test_two_sample(df_a, df_b, alpha=0.05, k=DEFAULT_K, n_bins=DEFAULT_BINS,
                    seed=0, max_t=None, ties=DEFAULT_TIES, bound="reply"):
    """H0: delta_A <= delta_B. Reject = 'A is more scripted than B'.

    Each system keeps its own streaming profile *and its own label alphabet*,
    so this compares how well each system's own accumulated routine predicts
    its own next reply — not whether one system's replies look like the
    other's. Each edge is already normalised against that system's own
    within-reply shuffles, so the paired difference is mean-zero under
    equality, which is all the betting construction needs.
    """
    reps_a, labs_a = _prepare(df_a, n_bins, seed, ties=ties)
    reps_b, labs_b = _prepare(df_b, n_bins, seed, ties=ties)
    prof_a = StreamingProfile(labs_a, n_bins)
    prof_b = StreamingProfile(labs_b, n_bins)
    rng = np.random.default_rng(seed + 1)
    bettor = Bettor(alpha, bound)
    n = min(len(reps_a), len(reps_b))
    if max_t:
        n = min(n, max_t)
    diffs = []
    for i in range(n):
        la, ba, ta = reps_a[i]
        lb, bb, tb = reps_b[i]
        d = prof_a.payoff_bound(la, ba, ta) + prof_b.payoff_bound(lb, bb, tb)
        bettor.stake(d)
        ea = edge(prof_a, la, ba, ta, rng, k)
        eb = edge(prof_b, lb, bb, tb, rng, k)
        diffs.append(ea - eb)
        bettor.observe(ea - eb, d)
        prof_a.update(la, ba, ta)
        prof_b.update(lb, bb, tb)
    diffs = np.array(diffs)
    return dict(rejected=bettor.rejected, tau=bettor.tau, n_pairs=n,
                wealth=bettor.wealth,
                diff_mean=float(diffs.mean()) if len(diffs) else np.nan,
                alpha=alpha, trace=bettor.trace)


def confidence_seq(df, alpha=0.05, grid=None, k=DEFAULT_K,
                   n_bins=DEFAULT_BINS, seed=0, max_t=None, ties=DEFAULT_TIES,
                   bound="reply"):
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
    replies, labels = _prepare(df, n_bins, seed, ties=ties)
    if max_t:
        replies = replies[:max_t]
    rng = np.random.default_rng(seed + 1)
    prof = StreamingProfile(labels, n_bins)
    bettors = {float(g): Bettor(alpha, bound) for g in grid}
    lows, highs = [], []
    for lab, bins, tmask in replies:
        d0 = prof.payoff_bound(lab, bins, tmask)
        for g, b in bettors.items():
            if not b.rejected:
                b.stake(d0 + abs(g))
        e = edge(prof, lab, bins, tmask, rng, k)
        alive = []
        for g, b in bettors.items():
            if not b.rejected:
                b.observe(e - g, d0 + abs(g))
            if not b.rejected:
                alive.append(g)
        prof.update(lab, bins, tmask)
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
    t.add_argument("--ties", choices=["order", "exclude"], default=DEFAULT_TIES)

    c = sub.add_parser("compare", help="sequentially test A more scripted than B")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--shuffles", type=int, default=DEFAULT_K)
    c.add_argument("--bins", type=int, default=DEFAULT_BINS)
    c.add_argument("--ties", choices=["order", "exclude"], default=DEFAULT_TIES)

    args = ap.parse_args()

    if args.cmd == "test":
        res = test_structure(load_spans(args.spans), alpha=args.alpha,
                             eps=args.eps, k=args.shuffles, n_bins=args.bins,
                             ties=args.ties)
        out = {k: v for k, v in res.items() if k != "trace"}
        out["verdict"] = ("templated: edge exceeds eps"
                          if res["rejected"] else "not established")
        if args.ci:
            cs = confidence_seq(load_spans(args.spans), alpha=args.alpha,
                                k=args.shuffles, n_bins=args.bins,
                                ties=args.ties)
            out["lower_confidence_bound_on_delta"] = round(
                float(cs["lower"][-1]), 4)
        print(json.dumps(out, indent=2, default=float))

    elif args.cmd == "compare":
        res = test_two_sample(load_spans(args.a), load_spans(args.b),
                              alpha=args.alpha, k=args.shuffles,
                              n_bins=args.bins, ties=args.ties)
        out = {k: v for k, v in res.items() if k != "trace"}
        out["verdict"] = ("A is more scripted than B" if res["rejected"]
                          else "no difference established")
        print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
