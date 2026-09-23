#!/usr/bin/env python3
"""Uncertainty and pairwise tests: paired cluster bootstrap over prompt items, estimator
variants (kNN mutual information, order-2 momentum), and reply-length terciles.

    python -m pipeline.metric.robustness BOOT:<start>:<stop>   one checkpointed bootstrap chunk
    python -m pipeline.metric.robustness COLLECT               merge chunks into CIs and Holm-corrected tests
    python -m pipeline.metric.robustness ALL                   the non-bootstrap checks

Writes tables/bootstrap_ci.csv, pairwise_tests.csv, estimator_variants.csv, length_terciles.csv."""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from pipeline.common.paths import CKPT, CORPORA, DATA_DIR, EVENTS, MODELS, TAB

from scriptmetric import metric as sm

NB = 10
rng = np.random.default_rng(11)

E = pd.read_csv(EVENTS)
E["unit"] = E.corpus + "|" + E.row.astype(str)
E["response_id"] = E.unit + "|" + E.model
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "ALL"

# =============================================== R1: paired cluster bootstrap
B_SHUF = 20
units = sorted(E.unit.unique())
# Pre-group events: per model, per unit -> (label idx, position bin) arrays
pre, lab_maps = {}, {}
for m in MODELS:
    d = E[E.model == m]
    labels = sorted(d.label.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab_maps[m] = labels
    g = {}
    for u, dd in d.groupby("unit"):
        dd = dd.sort_values("position", kind="stable")
        g[u] = (dd.label.map(l2i).to_numpy(),
                np.minimum((dd.position.to_numpy() * NB).astype(int), NB - 1),
                dd.position.to_numpy())
    pre[m] = g


def script_of(lab, xbin, rid, n_lab, n_shuf, rr, pos):
    """Excess (C, M, R) over a fresh within-response shuffle null.

    Positions are passed through so that the transition mask follows the slot
    convention of scriptmetric.metric.compute (co-located spans form no transition)."""
    C, M, R = sm._metrics(lab, xbin, rid, n_lab, NB, pos=pos, ties="exclude")
    null = np.array([sm._metrics(sm._shuffle_within(lab, rid, rr), xbin, rid,
                                 n_lab, NB, pos=pos, ties="exclude")
                     for _ in range(n_shuf)])
    mu = null.mean(0)
    return C - mu[0], M - mu[1], R - mu[2]


BOOT_CKPT = CKPT / "boot_ckpt"
BOOT_CKPT.mkdir(exist_ok=True)

if STAGE.startswith("BOOT"):
    b0, b1 = [int(v) for v in STAGE.split(":")[1:3]]
    print(f"== R1 bootstrap replicates {b0}-{b1} ==", flush=True)
    for b in range(b0, b1):
        rr = np.random.default_rng(1000 + b)   # replicate-specific, reproducible
        samp = rr.choice(units, size=len(units), replace=True)
        out = np.zeros((5, 3))
        for mi, m in enumerate(MODELS):
            g = pre[m]
            labs, xbs, rids, poss = [], [], [], []
            rid = 0
            for u in samp:
                arr = g.get(u)
                if arr is None or len(arr[0]) == 0:
                    continue
                labs.append(arr[0])
                xbs.append(arr[1])
                poss.append(arr[2])
                rids.append(np.full(len(arr[0]), rid))
                rid += 1
            lab = np.concatenate(labs)
            xb = np.concatenate(xbs)
            ri = np.concatenate(rids)
            ps = np.concatenate(poss)
            out[mi] = script_of(lab, xb, ri, len(lab_maps[m]), B_SHUF, rr, ps)
        np.save(BOOT_CKPT / f"rep{b:04d}.npy", out)
        if (b + 1) % 10 == 0:
            print(f"  rep {b + 1}", flush=True)
    print("chunk done")
    sys.exit(0)

if STAGE == "COLLECT":
    files = sorted(BOOT_CKPT.glob("rep*.npy"))
    print(f"collecting {len(files)} replicates", flush=True)
    A = np.stack([np.load(f) for f in files])   # (B, 5, 3)
    BT = {m: A[:, i, :] for i, m in enumerate(MODELS)}
    B_REPS = len(files)

    rows = []
    for m in MODELS:
        A = BT[m]
        r = dict(model=m)
        for j, nm in enumerate(["C", "M", "SCRIPT"]):
            r[nm + "_mean"] = round(A[:, j].mean(), 4)
            r[nm + "_lo"], r[nm + "_hi"] = [round(v, 4)
                                            for v in np.percentile(A[:, j], [2.5, 97.5])]
        rows.append(r)
    pd.DataFrame(rows).to_csv(TAB / "bootstrap_ci.csv", index=False)
    print(pd.DataFrame(rows)[["model", "SCRIPT_mean", "SCRIPT_lo", "SCRIPT_hi"]], flush=True)

    # Paired deltas with Holm-corrected two-sided bootstrap p-values.
    pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
    prow = []
    for a, b in pairs:
        D = BT[a][:, 2] - BT[b][:, 2]
        p = 2 * min((D <= 0).mean(), (D >= 0).mean())
        p = max(p, 1 / B_REPS)
        prow.append(dict(pair=f"{a}-{b}", delta=round(D.mean(), 4),
                         lo=round(np.percentile(D, 2.5), 4),
                         hi=round(np.percentile(D, 97.5), 4),
                         p_raw=round(p, 4)))
    PT = pd.DataFrame(prow).sort_values("p_raw").reset_index(drop=True)
    nP = len(PT)
    adj, run = [], 0.0
    for i in range(nP):
        run = max(run, (nP - i) * PT.p_raw[i])
        adj.append(round(min(1.0, run), 4))
    PT["p_holm"] = adj
    PT.to_csv(TAB / "pairwise_tests.csv", index=False)
    print(PT, flush=True)
    sys.exit(0)

# =============================================== R2a: kNN MI (Ross 2014) for C
print("== R2a kNN MI ==", flush=True)
try:
    from scipy.special import digamma as _dg
except Exception:
    def _dg(x):
        """Digamma via recurrence + asymptotic expansion (scipy fallback)."""
        x = np.asarray(x, float)
        r = np.zeros_like(x)
        y = x.copy()
        while np.any(y < 6):
            r[y < 6] -= 1 / y[y < 6]
            y[y < 6] += 1
        f = 1 / (y * y)
        r += np.log(y) - 0.5 / y - f * (1 / 12 - f * (1 / 120 - f / 252))
        return r


def knn_mi_cd(labels, x, k=3):
    """Ross (2014) MI between a discrete variable and a continuous one, in bits."""
    N = len(x)
    order = np.argsort(x)
    xs = x[order]
    ls = labels[order]
    m_all = np.zeros(N)
    k_all = np.zeros(N)
    nl_all = np.zeros(N)
    valid = np.zeros(N, bool)
    for lab in np.unique(ls):
        idx = np.flatnonzero(ls == lab)
        n = len(idx)
        if n < 2:
            continue
        kk = min(k, n - 1)
        arr = xs[idx]
        # distance to kk-th nearest neighbour within class (1D, vectorised)
        offs = [o for o in range(-kk, kk + 1) if o != 0]
        Dm = np.full((n, len(offs)), np.inf)
        for jj, o in enumerate(offs):
            src = np.arange(n) + o
            ok = (src >= 0) & (src < n)
            Dm[ok, jj] = np.abs(arr[ok] - arr[src[ok]])
        d = np.partition(Dm, kk - 1, axis=1)[:, kk - 1]
        # m_i: count of ALL points within that radius (inclusive)
        lo = np.searchsorted(xs, arr - d, side="left")
        hi = np.searchsorted(xs, arr + d, side="right")
        m = hi - lo - 1          # exclude self
        m_all[idx] = np.maximum(m, 1)
        k_all[idx] = kk
        nl_all[idx] = n
        valid[idx] = True
    v = valid
    mi = (_dg(np.array([v.sum()])).item() - _dg(nl_all[v]).mean()
          + _dg(k_all[v]).mean() - _dg(m_all[v]).mean())
    return mi / np.log(2)


est_rows = []
for m in MODELS:
    d = E[E.model == m].sort_values(["response_id", "position"])
    labels = pd.factorize(d.label)[0]
    x = d.position.to_numpy() + rng.normal(0, 1e-9, len(d))   # break exact ties
    counts = np.bincount(labels).astype(float)
    H_L = -(counts / counts.sum() * np.log2(counts / counts.sum())).sum()
    rid = pd.factorize(d.response_id)[0]
    C_knn = knn_mi_cd(labels, x) / H_L
    null = []
    for _ in range(60):
        null.append(knn_mi_cd(sm._shuffle_within(labels, rid, rng), x) / H_L)
    null = np.array(null)
    est_rows.append(dict(model=m, C_knn_excess=round(C_knn - null.mean(), 4),
                         z=round((C_knn - null.mean()) / null.std(), 1)))
    print(est_rows[-1], flush=True)

# =============================================== R2b: 2nd-order momentum (groups)
print("== R2b order-2 momentum ==", flush=True)
GROUPS = {"emp_acc": ["VAC", "NAC", "ASAC", "SAC"],
          "emp_in": ["VIN", "NIN", "ASIN", "SIN"],
          "advice": ["DIR", "FIX", "RECT"],
          "quest": ["QOP", "QCL"],
          "other": ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}
C2G = {c: g for g, cs in GROUPS.items() for c in cs}
GL = list(GROUPS)
NG = len(GL)


def slot_predecessor(rid, pos):
    """Index of each event's predecessor under the slot convention: the last
    event at the preceding *distinct* position in the same reply (-1 if none).
    Depends on positions only, so it is shared by the data and every shuffle."""
    prev = np.full(len(rid), -1)
    for i in range(1, len(rid)):
        j = i - 1
        while j >= 0 and rid[j] == rid[i] and pos[j] == pos[i]:
            j -= 1
        if j >= 0 and rid[j] == rid[i]:
            prev[i] = j
    return prev


def order2_gain(lab, xbin, rid, prev):
    """I(L; L_prev2 | X, L_prev) / H(L) at behaviour-group level, on events
    with two slot-convention predecessors."""
    counts = np.bincount(lab, minlength=NG).astype(float)
    p = counts[counts > 0] / counts.sum()
    H_L = -(p * np.log2(p)).sum()
    idx = np.flatnonzero((prev >= 0) & (prev[np.maximum(prev, 0)] >= 0))
    i1 = prev[idx]
    i2 = prev[i1]
    cur, p1, p2, xb = lab[idx], lab[i1], lab[i2], xbin[idx]
    n = len(cur)
    if n == 0:
        return 0.0
    tot = 0.0
    for b in range(NB):
        for pv in range(NG):
            msk = (xb == b) & (p1 == pv)
            if msk.sum() < 2:
                continue
            J = np.zeros((NG, NG))
            np.add.at(J, (cur[msk], p2[msk]), 1.0)
            tot += (msk.sum() / n) * (sm._H(J.sum(1)) + sm._H(J.sum(0)) - sm._H(J.ravel()))
    return tot / H_L


for i, m in enumerate(MODELS):
    d = E[E.model == m].sort_values(["response_id", "position"])
    lab = d.label.map(C2G).map({g: i for i, g in enumerate(GL)}).to_numpy()
    xb = np.minimum((d.position.to_numpy() * NB).astype(int), NB - 1)
    rid = pd.factorize(d.response_id)[0]
    prev = slot_predecessor(rid, d.position.to_numpy())
    g2 = order2_gain(lab, xb, rid, prev)
    null = np.array([order2_gain(sm._shuffle_within(lab, rid, rng), xb, rid, prev)
                     for _ in range(60)])
    est_rows[i]["M2gain_excess"] = round(g2 - null.mean(), 4)
    est_rows[i]["M2_z"] = round((g2 - null.mean()) / null.std(), 1)
    print(m, est_rows[i]["M2gain_excess"], est_rows[i]["M2_z"], flush=True)
pd.DataFrame(est_rows).to_csv(TAB / "estimator_variants.csv", index=False)

# =============================================== R3: length terciles
print("== R3 length terciles ==", flush=True)


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("—", "--").replace("–", "-")
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    return re.sub(r"\s+", " ", s).strip().lower()


lens = {}
for corpus in CORPORA:
    df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    for i, r in df.iterrows():
        for m in MODELS:
            out = r.get(f"{m} Output")
            if pd.notna(out):
                lens[(corpus, i, m)] = len(norm(out))
E["rlen"] = [lens.get((c, rw, m), np.nan) for c, rw, m in zip(E.corpus, E.row, E.model)]
len_rows = []
for m in MODELS:
    d = E[E.model == m]
    per_resp = d.groupby("response_id").rlen.first()
    q1, q2 = per_resp.quantile([1 / 3, 2 / 3])
    terc = {"short": per_resp[per_resp <= q1].index,
            "mid": per_resp[(per_resp > q1) & (per_resp <= q2)].index,
            "long": per_resp[per_resp > q2].index}
    for t, ids in terc.items():
        dd = d[d.response_id.isin(ids)][["response_id", "label", "position"]]
        res, _ = sm.compute(dd.sort_values(["response_id", "position"]).reset_index(drop=True),
                            n_bins=NB, n_shuffles=60, seed=0)
        len_rows.append(dict(model=m, tercile=t, SCRIPT=res["SCRIPT"], z=res["z"],
                             n_events=res["n_events"], n_resp=res["n_responses"]))
        print(len_rows[-1], flush=True)
pd.DataFrame(len_rows).to_csv(TAB / "length_terciles.csv", index=False)

# RAGTruth and the other external-scheme anchors live in
# external/anchors.py, which recomputes all three from their sources.

print("ALL DONE")
