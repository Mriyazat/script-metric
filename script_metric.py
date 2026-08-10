#!/usr/bin/env python3
"""
SCRIPT — Structural Choreography & Rigidity Index from Positions and
Transitions: a reference-free metric of behavioral scriptedness for LLMs,
computed from span annotations alone.

    SCRIPT score = share of a model's annotated-behavior uncertainty that is
    explained by the response's internal STRUCTURE (slot + previous move)
    rather than by its content — measured in excess of chance.

    R = C + M
    C (choreography) = I(L ; X)        / H(L)   "labels have fixed slots"
    M (momentum)     = I(L ; L_prev|X) / H(L)   "labels follow each other"
    SCRIPT = R_observed − mean(R_shuffled)      (within-response label shuffle)
    z      = (R_observed − mean) / sd           (significance vs chance)

Input: a CSV of span annotations for ONE system. Required columns, either
    response_id, label, position            (position in [0,1]), or
    response_id, label, start, response_length   (position = start/length)
Any label scheme works (behavior codes, error types, toxicity spans, ...).
Nothing else is needed: no references, no user ratings, no second model.

The by-product is the system's SCRIPT PROFILE — its (label x slot) and
(previous x current label) distributions. The profile is the system's
behavioral fingerprint: the Jensen-Shannon distance between two profiles is
a behavioral distance, and nearest-profile matching identifies which system
produced an unlabeled set of annotated responses.

Usage
-----
    python3 script_metric.py score  spans.csv  [--shuffles 200] [--bins 10]
                                               [--profile out.json]
                                               [--ceiling] [--extrapolate]
    python3 script_metric.py distance  a.json  b.json
    python3 script_metric.py identify  probe_spans.csv  profiles_dir/

--ceiling      also reports the MATCHED CEILING: the score of the most
               scripted arrangement of this corpus's own events (same
               responses, same positions, same per-response label multisets,
               labels re-dealt in a fixed position order). SCRIPT / ceiling
               is the fraction of achievable rigidity — a [0,1] quantity
               that travels across annotation schemes better than the raw
               score, whose ceiling depends on the alphabet and event density.
--extrapolate  also reports a small-sample bias-corrected estimate:
               subsample the corpus, fit SCRIPT against 1/n_events, report
               the extrapolated asymptote. Recovers the large-sample value
               from roughly 200 responses; below that it reduces the
               downward bias but does not remove it (report z there).

Dependencies: numpy, pandas. Single file; no other code needed.
Interpretation ruler (measured, three unrelated annotation schemes):
    ~0.00        content-driven, no template   (WMT24 MT error spans)
    ~0.05        partial structure, streaks    (data-to-text error spans)
    ~0.10 z≈40   hard behavioral template      (LLM counseling behavior codes)
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- internals

def _H(counts):
    p = counts[counts > 0].astype(float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _metrics(lab, xbin, rid, n_lab, n_bins):
    """C, M, R from integer arrays sorted by (rid, position)."""
    H_L = _H(np.bincount(lab, minlength=n_lab).astype(float))
    if H_L == 0:
        return 0.0, 0.0, 0.0
    joint = np.zeros((n_lab, n_bins))
    np.add.at(joint, (lab, xbin), 1.0)
    I_LX = _H(joint.sum(1)) + _H(joint.sum(0)) - _H(joint.ravel())
    same = rid[1:] == rid[:-1]
    cur, prev, xb = lab[1:][same], lab[:-1][same], xbin[1:][same]
    I_cond, n_tr = 0.0, max(len(cur), 1)
    for b in range(n_bins):
        m = xb == b
        if m.sum() < 2:
            continue
        j2 = np.zeros((n_lab, n_lab))
        np.add.at(j2, (cur[m], prev[m]), 1.0)
        I_cond += (m.sum() / n_tr) * (_H(j2.sum(1)) + _H(j2.sum(0)) - _H(j2.ravel()))
    return I_LX / H_L, I_cond / H_L, I_LX / H_L + I_cond / H_L


def _shuffle_within(lab, rid, rng):
    out = lab.copy()
    order = np.argsort(rid, kind="stable")
    bounds = np.flatnonzero(np.diff(rid[order])) + 1
    for seg in np.split(order, bounds):
        out[seg] = out[rng.permutation(seg)]
    return out


def load_spans(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"response_id", "label"}
    if not need <= set(df.columns):
        sys.exit(f"spans file must have columns {need} plus position "
                 f"(or start + response_length); got {list(df.columns)}")
    if "position" not in df.columns:
        if {"start", "response_length"} <= set(df.columns):
            df["position"] = df["start"] / df["response_length"]
        else:
            sys.exit("need either 'position' or 'start'+'response_length'")
    df = df.dropna(subset=["position", "label"])
    df = df[(df.position >= 0) & (df.position <= 1)]
    return df.sort_values(["response_id", "position"]).reset_index(drop=True)


def compute(df, n_bins=10, n_shuffles=200, seed=0):
    labels = sorted(df.label.astype(str).unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab = df.label.astype(str).map(l2i).to_numpy()
    xbin = np.minimum((df.position.to_numpy() * n_bins).astype(int), n_bins - 1)
    rid = pd.factorize(df.response_id)[0]
    C, M, R = _metrics(lab, xbin, rid, len(labels), n_bins)
    rng = np.random.default_rng(seed)
    null = np.array([_metrics(_shuffle_within(lab, rid, rng), xbin, rid,
                              len(labels), n_bins) for _ in range(n_shuffles)])
    mu, sd = null.mean(0), null.std(0)
    # profile = the fingerprint (unsmoothed counts; smooth at comparison time)
    pos = np.zeros((len(labels), n_bins))
    np.add.at(pos, (lab, xbin), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return dict(
        SCRIPT=round(R - mu[2], 4),
        z=round(float((R - mu[2]) / sd[2]) if sd[2] > 0 else float("nan"), 2),
        C_excess=round(C - mu[0], 4), M_excess=round(M - mu[1], 4),
        R_raw=round(R, 4), R_null=round(float(mu[2]), 4),
        n_events=int(len(df)), n_responses=int(df.response_id.nunique()),
        n_labels=len(labels), n_bins=n_bins, n_shuffles=n_shuffles,
    ), dict(labels=labels, position=pos.tolist(), transition=tr.tolist(),
            n_bins=n_bins)


# --------------------------------------------- ceiling / extrapolation add-ons

def matched_ceiling(df, n_bins=10, n_shuffles=200, seed=0):
    """SCRIPT of the most scripted arrangement of this corpus's own events.

    Deterministic construction inside the null's invariance class: keep every
    response, every position, and every response's own label multiset, and
    re-deal each response's labels in a single canonical order (labels sorted
    by their global mean position). Labels then occupy fixed slots and chain
    in blocks — the ceiling the data's own composition and density allow.

    SCRIPT / ceiling is the "fraction of achievable rigidity": 0 = content-
    driven, 1 = as scripted as this corpus could possibly be. Unlike the raw
    score, whose ceiling varies with alphabet size and events per response,
    the fraction is read on one scale across annotation schemes.
    """
    order = {l: i for i, l in
             enumerate(df.groupby(df.label.astype(str)).position.mean()
                       .sort_values().index)}
    parts = []
    for _, g in df.groupby("response_id", sort=False):
        g = g.sort_values("position").copy()
        g["label"] = sorted(g.label.astype(str), key=lambda l: order[l])
        parts.append(g)
    arranged = pd.concat(parts, ignore_index=True)
    res, _ = compute(arranged, n_bins=n_bins, n_shuffles=n_shuffles, seed=seed)
    return res


def extrapolate(df, n_bins=10, fractions=(0.25, 0.4, 0.6, 0.8, 1.0),
                draws=10, n_shuffles=60, seed=0):
    """Small-sample bias-corrected SCRIPT by subsample extrapolation.

    The null-subtracted score is downward-biased at small n (the signal has
    not risen above the noise floor the null removes). This estimator
    subsamples responses at several fractions, fits SCRIPT against
    1/n_events, and reports the intercept — the value the corpus is heading
    toward. Validated on the benchmark: recovers the full-sample score from
    ~200 responses; below that it shrinks the bias but stays conservative.
    """
    rng = np.random.default_rng(seed)
    rids = df.response_id.unique()
    intercepts, curve = [], {f: [] for f in fractions}
    for d in range(draws):
        xs, ys = [], []
        for f in fractions:
            k = max(10, int(round(len(rids) * f)))
            ids = rng.choice(rids, min(k, len(rids)), replace=False)
            r, _ = compute(df[df.response_id.isin(ids)], n_bins=n_bins,
                           n_shuffles=n_shuffles, seed=seed * 1000 + d)
            xs.append(1.0 / r["n_events"]); ys.append(r["SCRIPT"])
            curve[f].append((r["n_events"], r["SCRIPT"]))
        intercepts.append(float(np.polyfit(xs, ys, 1)[-1]))
    pts = [(int(np.mean([a for a, _ in curve[f]])),
            round(float(np.mean([b for _, b in curve[f]])), 4))
           for f in fractions]
    return dict(SCRIPT_extrapolated=round(float(np.mean(intercepts)), 4),
                se=round(float(np.std(intercepts)), 4),
                subsample_curve=pts, draws=draws)


# ------------------------------------------------- profile distance / identify

def _js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _align(pr, labels):
    """Re-index a profile onto a shared label list (missing labels = 0)."""
    idx = {l: i for i, l in enumerate(pr["labels"])}
    nb = pr["n_bins"]
    pos = np.zeros((len(labels), nb)); tr = np.zeros((len(labels), len(labels)))
    P, T = np.array(pr["position"]), np.array(pr["transition"])
    for a, la in enumerate(labels):
        if la not in idx:
            continue
        pos[a] = P[idx[la]]
        for b, lb in enumerate(labels):
            if lb in idx:
                tr[a, b] = T[idx[la], idx[lb]]
    return pos, tr


def profile_distance(pa, pb, alpha=0.5):
    """JS distance between two profiles. Use for SYSTEM-to-SYSTEM comparison
    (both profiles built from many responses). For identifying a small probe
    against references, use profile_loglik instead — JS on a sparse probe is
    dominated by smoothing and gravitates to the flattest reference."""
    labels = sorted(set(pa["labels"]) | set(pb["labels"]))
    Pa, Ta = _align(pa, labels); Pb, Tb = _align(pb, labels)
    d_pos = _js(Pa.ravel() + alpha, Pb.ravel() + alpha)
    d_tr = _js(Ta.ravel() + alpha, Tb.ravel() + alpha)
    return 0.5 * (d_pos + d_tr)


def profile_loglik(probe_df, ref, alpha=0.5):
    """Mean per-event log-likelihood of probe span events under a reference
    profile: log P_ref(label | position bin) + log P_ref(label | prev label).
    Works at any probe size (even a single response); higher = better match."""
    labs = ref["labels"]; l2i = {l: i for i, l in enumerate(labs)}
    nb = ref["n_bins"]
    P = np.array(ref["position"]) + alpha
    T = np.array(ref["transition"]) + alpha
    P = P / P.sum(0, keepdims=True)          # P(label | bin)
    T = T / T.sum(1, keepdims=True)          # P(current | previous)
    tot, n = 0.0, 0
    prev = {}
    d = probe_df.sort_values(["response_id", "position"])
    for x in d.itertuples():
        b = min(int(x.position * nb), nb - 1)
        li = l2i.get(str(x.label))
        tot += np.log(P[li, b] if li is not None else 1.0 / len(labs))
        n += 1
        pr = prev.get(x.response_id)
        if pr is not None and pr >= 0 and li is not None:
            tot += np.log(T[pr, li])
        prev[x.response_id] = li if li is not None else -1
    return tot / max(n, 1)


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="compute the SCRIPT score of one system")
    s.add_argument("spans"); s.add_argument("--bins", type=int, default=10)
    s.add_argument("--shuffles", type=int, default=200)
    s.add_argument("--profile", help="write the fingerprint profile JSON here")
    s.add_argument("--ceiling", action="store_true",
                   help="also report the matched ceiling and SCRIPT/ceiling")
    s.add_argument("--extrapolate", action="store_true",
                   help="also report the small-sample bias-corrected estimate")
    d = sub.add_parser("distance", help="behavioral distance between two profiles")
    d.add_argument("a"); d.add_argument("b")
    i = sub.add_parser("identify", help="match probe spans to stored profiles")
    i.add_argument("spans"); i.add_argument("profiles_dir")
    i.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "score":
        df = load_spans(args.spans)
        res, prof = compute(df, args.bins, args.shuffles)
        if args.ceiling:
            ceil = matched_ceiling(df, args.bins, args.shuffles)
            res["ceiling"] = max(ceil["SCRIPT"], res["SCRIPT"])
            res["fraction_of_ceiling"] = round(
                res["SCRIPT"] / res["ceiling"], 4) if res["ceiling"] > 0 \
                else float("nan")
        if args.extrapolate:
            res.update(extrapolate(df, args.bins))
        print(json.dumps(res, indent=2))
        if args.profile:
            with open(args.profile, "w") as f:
                json.dump(prof, f)
            print(f"profile -> {args.profile}", file=sys.stderr)

    elif args.cmd == "distance":
        pa, pb = (json.load(open(p)) for p in (args.a, args.b))
        print(json.dumps({"JS_distance": round(profile_distance(pa, pb), 4)}))

    elif args.cmd == "identify":
        probe_df = load_spans(args.spans)
        scores = {}
        for f in sorted(glob.glob(os.path.join(args.profiles_dir, "*.json"))):
            scores[os.path.splitext(os.path.basename(f))[0]] = \
                profile_loglik(probe_df, json.load(open(f)))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("nan")
        print(json.dumps({"identified": ranked[0][0],
                          "margin_to_runner_up": round(margin, 4),
                          "mean_loglik_per_event": {k: round(v, 4)
                                                    for k, v in ranked}},
                         indent=2))


if __name__ == "__main__":
    main()
