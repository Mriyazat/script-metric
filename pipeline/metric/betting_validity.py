#!/usr/bin/env python3
"""SCRIPT-Seq, the test by betting: validity, power and scaling.

    V1  false-rejection rate under the null, against naive repeated z-testing
    V2  annotated replies needed to certify each system's template
    V3  stopping time against 1/delta^2

    python -m pipeline.metric.betting_validity [V1|V2|V3|ALL] [--streams 200]

Writes tables/betting_validity.csv, betting_power.csv, betting_scaling.csv."""
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import CORPORA, EVENTS, MODELS, TAB

from scriptmetric import betting as sb
from scriptmetric import metric as sm

ALPHA = 0.05
MAX_T = 400
N_BINS = 10


def load_events() -> pd.DataFrame:
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    return E.sort_values(["response_id", "position"]).reset_index(drop=True)


def h0_version(d: pd.DataFrame, rng) -> pd.DataFrame:
    """Materialise the null: permute labels within each reply.

    Behaviour mix, reply length and every position survive; only the
    label-to-position pairing is destroyed. Any rejection on this data is by
    definition a false positive.
    """
    out = d.copy()
    out["label"] = out.groupby("response_id").label.transform(
        lambda s: rng.permutation(s.to_numpy()))
    return out


def naive_peek(d: pd.DataFrame, seed=0, every=20, z_crit=1.96, n_shuffles=30,
               max_t=MAX_T):
    """Recompute the published z as replies arrive; stop when it clears 1.96.

    This is what a practitioner would naturally do with the metric as
    published, and it has no error guarantee at all.
    """
    rng = np.random.default_rng(seed)
    order = d.response_id.unique()
    labels = sorted(d.label.astype(str).unique())
    idx = {l: i for i, l in enumerate(labels)}
    groups = {r: g for r, g in d.groupby("response_id", sort=False)}
    lab_acc, bin_acc, rid_acc, pos_acc = [], [], [], []
    for t, r in enumerate(order[:max_t], start=1):
        g = groups[r].sort_values("position", kind="stable")
        lab_acc.append(np.array([idx[str(x)] for x in g.label]))
        bin_acc.append(np.minimum((g.position.to_numpy() * N_BINS).astype(int),
                                  N_BINS - 1))
        pos_acc.append(g.position.to_numpy())
        rid_acc.append(np.full(len(g), t))
        if t % every or t < 20:
            continue
        lab = np.concatenate(lab_acc)
        xb = np.concatenate(bin_acc)
        rid = np.concatenate(rid_acc)
        ps = np.concatenate(pos_acc)
        # slot convention, as in scriptmetric.metric.compute
        _, _, R = sm._metrics(lab, xb, rid, len(labels), N_BINS, pos=ps, ties="exclude")
        null = np.array([sm._metrics(sm._shuffle_within(lab, rid, rng), xb, rid,
                                     len(labels), N_BINS, pos=ps, ties="exclude")[2]
                         for _ in range(n_shuffles)])
        sd = null.std()
        if sd > 0 and (R - null.mean()) / sd >= z_crit:
            return True, t
    return False, None


# ------------------------------------------------------------------------ V1

def v1(E: pd.DataFrame, streams: int, naive_streams: int) -> pd.DataFrame:
    print("=" * 72)
    print(f"V1 — type-I error under H0   (alpha = {ALPHA}, {streams} streams, "
          f"up to {MAX_T} replies)")
    print("=" * 72)
    base = E[E.model == "Claude"][["response_id", "label", "position"]]
    rows = []
    rej = 0
    for s in range(streams):
        rng = np.random.default_rng(1000 + s)
        r = sb.test_structure(h0_version(base, rng), alpha=ALPHA, seed=s,
                              max_t=MAX_T)
        rej += r["rejected"]
    print(f"  betting (Ville)              {rej}/{streams} = "
          f"{rej/streams:.3f}    guarantee <= {ALPHA}")
    rows.append(dict(method="betting (anytime-valid)", rejections=rej,
                     streams=streams, rate=rej / streams, guarantee=ALPHA))

    if naive_streams:
        nrej = 0
        for s in range(naive_streams):
            rng = np.random.default_rng(1000 + s)
            ok, _ = naive_peek(h0_version(base, rng), seed=s)
            nrej += ok
        print(f"  naive: peek at z every 20    {nrej}/{naive_streams} = "
              f"{nrej/naive_streams:.3f}    no guarantee")
        rows.append(dict(method="naive z peeking", rejections=nrej,
                         streams=naive_streams, rate=nrej / naive_streams,
                         guarantee=np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "betting_validity.csv", index=False)
    print(f"\n  wrote {TAB / 'betting_validity.csv'}")
    return T


# ------------------------------------------------------------------------ V2

def v2(E: pd.DataFrame, reps: int = 30) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print(f"V2 — power: replies needed to declare, alpha = {ALPHA}")
    print("=" * 72)
    rows = []

    print("\n  pooled, per model:")
    for m in MODELS:
        d = E[E.model == m][["response_id", "label", "position"]]
        taus = []
        for s in range(reps):
            r = sb.test_structure(d, alpha=ALPHA, seed=s, max_t=MAX_T)
            if r["rejected"]:
                taus.append(r["tau"])
        rows.append(dict(scope="pooled", unit=m, detected=len(taus),
                         streams=reps,
                         median_tau=int(np.median(taus)) if taus else np.nan,
                         q25=int(np.percentile(taus, 25)) if taus else np.nan,
                         q75=int(np.percentile(taus, 75)) if taus else np.nan))
        print(f"    {m:8s} {len(taus)}/{reps} detected   median "
              f"{rows[-1]['median_tau']} replies   "
              f"IQR [{rows[-1]['q25']}, {rows[-1]['q75']}]")

    print("\n  per corpus (Claude), the harder small-sample setting:")
    for c in CORPORA:
        d = E[(E.model == "Claude") & (E.corpus == c)][
            ["response_id", "label", "position"]]
        taus = []
        for s in range(reps):
            r = sb.test_structure(d, alpha=ALPHA, seed=s, max_t=MAX_T)
            if r["rejected"]:
                taus.append(r["tau"])
        n_resp = d.response_id.nunique()
        rows.append(dict(scope="corpus", unit=c, detected=len(taus),
                         streams=reps,
                         median_tau=int(np.median(taus)) if taus else np.nan,
                         q25=int(np.percentile(taus, 25)) if taus else np.nan,
                         q75=int(np.percentile(taus, 75)) if taus else np.nan))
        print(f"    {c:12s} {len(taus)}/{reps} detected   median "
              f"{rows[-1]['median_tau']}   (corpus has only {n_resp} replies)")

    print("\n  per annotator (pooled models), the annotator-confound setting:")
    for rv in sorted([r for r in E.reviewer.dropna().unique()
                      if str(r).startswith("R")]):
        d = E[E.reviewer == rv][["response_id", "label", "position"]]
        if d.response_id.nunique() < 30:
            continue
        taus = []
        for s in range(10):
            r = sb.test_structure(d, alpha=ALPHA, seed=s, max_t=MAX_T)
            if r["rejected"]:
                taus.append(r["tau"])
        rows.append(dict(scope="annotator", unit=rv, detected=len(taus),
                         streams=10,
                         median_tau=int(np.median(taus)) if taus else np.nan,
                         q25=np.nan, q75=np.nan))
        print(f"    {rv} {len(taus)}/10 detected   median "
              f"{rows[-1]['median_tau']}   ({d.response_id.nunique()} replies)")

    T = pd.DataFrame(rows)
    T.to_csv(TAB / "betting_power.csv", index=False)
    print(f"\n  wrote {TAB / 'betting_power.csv'}")
    return T


# ------------------------------------------------------------------------ V3

def dilute(d: pd.DataFrame, p: float, rng) -> pd.DataFrame:
    """Shuffle a fraction p of the replies. p=0 is the real corpus, p=1 is H0.

    This is the only way to get a controlled range of signal strengths out of
    one dataset: the real systems all happen to have similar edges, so they
    cannot trace a power curve. Diluting interpolates continuously between a
    fully templated system and a content-driven one, which is exactly the
    intermediate regime a deployment threshold has to discriminate.
    """
    out = d.copy()
    rids = out.response_id.unique()
    hit = set(rids[rng.random(len(rids)) < p])
    mask = out.response_id.isin(hit)
    if mask.any():
        out.loc[mask, "label"] = (out[mask].groupby("response_id").label
                                  .transform(lambda s: rng.permutation(s.to_numpy())))
    return out


def v3(E: pd.DataFrame, reps: int = 8) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("V3 — sensitivity: how weak a template can it still catch?")
    print("=" * 72)
    print("A fraction p of replies has its labels shuffled, so p = 0 is the")
    print("real corpus and p = 1 is the null. Everything in between is a")
    print("partially templated system — the case a deployment threshold has")
    print("to get right, and the case no real corpus here supplies.\n")
    base = E[E.model == "Claude"][["response_id", "label", "position"]]
    rows = []
    for p in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        taus, edges, hits = [], [], 0
        for s in range(reps):
            rng = np.random.default_rng(500 + s)
            r = sb.test_structure(dilute(base, p, rng), alpha=ALPHA, seed=s,
                                  max_t=MAX_T)
            edges.append(r["edge_mean"])
            hits += r["rejected"]
            if r["rejected"]:
                taus.append(r["tau"])
        rows.append(dict(shuffled_fraction=p, signal_kept=round(1 - p, 2),
                         edge=round(float(np.mean(edges)), 4),
                         detected=hits, streams=reps,
                         median_tau=int(np.median(taus)) if taus else np.nan))
        print(f"  p={p:<5} signal {1-p:>4.0%}   edge {rows[-1]['edge']:+.4f}   "
              f"detected {hits}/{reps}   median tau "
              f"{rows[-1]['median_tau']}")
    T = pd.DataFrame(rows)

    det = T[(T.detected == T.streams) & (T.shuffled_fraction < 1)]
    if len(det) > 2:
        x, y = np.log(det.edge.clip(lower=1e-6)), np.log(det.median_tau)
        slope = np.polyfit(x, y, 1)[0]
        print(f"\n  over the fully-detected range, log(tau) vs log(edge) "
              f"slope = {slope:.2f}")
        print("  (Chen & Wang's bound is O(1/delta^2), i.e. slope -2 in the "
              "regime\n   where the stopping time is signal-limited rather "
              "than burn-in-limited)")
    weakest = T[T.detected == T.streams].signal_kept.min()
    print(f"\n  weakest template still detected in every stream: "
          f"{weakest:.0%} of the original signal")
    T.to_csv(TAB / "betting_scaling.csv", index=False)
    print(f"\n  wrote {TAB / 'betting_scaling.csv'}")
    return T


def main() -> None:
    stage = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    streams = 200
    naive = 60
    for a in sys.argv[2:]:
        if a.startswith("--streams"):
            streams = int(a.split("=")[1])
        if a.startswith("--naive"):
            naive = int(a.split("=")[1])
    E = load_events()
    if stage in ("V1", "ALL"):
        v1(E, streams, naive)
    if stage in ("V2", "ALL"):
        v2(E)
    if stage in ("V3", "ALL"):
        v3(E)


if __name__ == "__main__":
    main()
