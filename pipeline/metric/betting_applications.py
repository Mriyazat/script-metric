#!/usr/bin/env python3
"""SCRIPT-Seq in use.

    A1  calibration of the betting edge to the SCRIPT scale
    A2  deployment threshold audit (composite null delta <= eps)
    A3  sequential A/B comparison of two systems on paired replies
    A4  transfer to the external span corpora and AnnoMI

    python -m pipeline.metric.betting_applications [A1|A2|A3|A4|ALL]

Writes tables/betting_calibration.csv, betting_threshold.csv, betting_ab.csv, betting_transfer.csv."""
import itertools
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import EVENTS, MODELS, RAW, TAB
from pipeline.external import anchors as anc

from scriptmetric import betting as sb
from scriptmetric import metric as sm

ALPHA = 0.05
MAX_T = 500
N_SHUFFLES = 200


def load_events() -> pd.DataFrame:
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    return E.sort_values(["response_id", "position"]).reset_index(drop=True)


def dilute(d, p, rng):
    out = d.copy()
    rids = out.response_id.unique()
    hit = set(rids[rng.random(len(rids)) < p])
    mask = out.response_id.isin(hit)
    if mask.any():
        out.loc[mask, "label"] = (
            out[mask].groupby("response_id").label
            .transform(lambda s: rng.permutation(s.to_numpy())))
    return out


def both(d, seed=0, max_t=MAX_T):
    """-> (SCRIPT result, betting result) for the same slice of data."""
    dd = (d[["response_id", "label", "position"]]
          .sort_values(["response_id", "position"]).reset_index(drop=True))
    res, _ = sm.compute(dd, n_bins=10, n_shuffles=N_SHUFFLES, seed=seed)
    bet = sb.test_structure(dd, alpha=ALPHA, seed=seed, max_t=max_t)
    return res, bet


# ------------------------------------------------------------------------ A1

def a1(E: pd.DataFrame) -> dict:
    print("=" * 72)
    print("A1 — calibrating the betting edge against SCRIPT")
    print("=" * 72)
    print("The two are different quantities: SCRIPT is a corpus-level")
    print("normalised mutual information, the edge is a per-reply predictive")
    print("gain. A dilution sweep moves both together over the full range,")
    print("which is what makes the map estimable.\n")
    base = E[E.model == "Claude"][["response_id", "label", "position"]]
    rows = []
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
        for s in range(2):
            rng = np.random.default_rng(900 + s)
            res, bet = both(dilute(base, p, rng), seed=s)
            rows.append(dict(source=f"Claude diluted p={p}",
                             SCRIPT=res["SCRIPT"], z=res["z"],
                             edge=bet["edge_mean"]))
    for m in MODELS:
        res, bet = both(E[E.model == m])
        rows.append(dict(source=m, SCRIPT=res["SCRIPT"], z=res["z"],
                         edge=bet["edge_mean"]))
    T = pd.DataFrame(rows)
    cal = sb.calibrate(T.edge, T.SCRIPT)
    T.to_csv(TAB / "betting_calibration.csv", index=False)
    print(f"  SCRIPT = {cal['slope']:.4f} * edge + {cal['intercept']:+.4f}"
          f"   (r = {cal['r']:.3f}, n = {cal['n']})")
    for ceiling in (0.02, 0.05, 0.08):
        print(f"    ceiling SCRIPT = {ceiling:.2f}  ->  eps = "
              f"{cal['to_eps'](ceiling):.3f} nats/event")
    print(f"\n  wrote {TAB / 'betting_calibration.csv'}")
    return cal


# ------------------------------------------------------------------------ A2

def a2(E: pd.DataFrame, cal: dict) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("A2 — the deployment audit: does this system exceed a ceiling?")
    print("=" * 72)
    print("Composite null H0: delta <= eps. Rejecting means the system sits")
    print("*above* the ceiling, with a false-alarm rate bounded at every look.")
    print("A system merely showing some structure no longer trips the alarm.\n")
    rows = []
    for ceiling in (0.02, 0.05, 0.08):
        eps = cal["to_eps"](ceiling)
        line = []
        for m in MODELS:
            d = E[E.model == m][["response_id", "label", "position"]]
            taus = []
            for s in range(6):
                r = sb.test_structure(d, alpha=ALPHA, eps=eps, seed=s,
                                      max_t=MAX_T)
                if r["rejected"]:
                    taus.append(r["tau"])
            rows.append(dict(ceiling_SCRIPT=ceiling, eps=round(eps, 3),
                             model=m, detected=len(taus), streams=6,
                             median_tau=int(np.median(taus)) if taus else np.nan))
            line.append(f"{m} {len(taus)}/6"
                        + (f"@{int(np.median(taus))}" if taus else ""))
        print(f"  ceiling {ceiling:.2f} (eps {eps:.3f}):  " + "  ".join(line))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "betting_threshold.csv", index=False)
    print(f"\n  wrote {TAB / 'betting_threshold.csv'}")
    return T


# ------------------------------------------------------------------------ A3

def a3(E: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("A3 — the regression test: is A more scripted than B?")
    print("=" * 72)
    print("Paired sequential comparison. The published analysis needed a")
    print("cluster bootstrap over the whole corpus and Holm correction across")
    print("ten pairs; this decides on the fly and stays valid however long it")
    print("runs.\n")
    rows = []
    for a, b in itertools.permutations(MODELS, 2):
        da = E[E.model == a][["response_id", "label", "position"]]
        db = E[E.model == b][["response_id", "label", "position"]]
        r = sb.test_two_sample(da, db, alpha=ALPHA, seed=0, max_t=MAX_T)
        rows.append(dict(A=a, B=b, rejected=r["rejected"], tau=r["tau"],
                         diff_mean=round(r["diff_mean"], 4),
                         wealth=r["wealth"]))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "betting_ab.csv", index=False)
    hits = T[T.rejected]
    print(f"  {len(hits)}/{len(T)} ordered pairs decided at alpha = {ALPHA}:")
    for _, r in hits.sort_values("tau").iterrows():
        print(f"    {r.A:7s} > {r.B:<7s} at t = {int(r.tau):3d} replies "
              f"(mean edge gap {r.diff_mean:+.3f})")
    if len(hits) == 0:
        print("    none")
    pairs = {"/".join(sorted([r.A, r.B])) for _, r in T[~T.rejected].iterrows()}
    decided = {"/".join(sorted([r.A, r.B])) for _, r in hits.iterrows()}
    print(f"\n  never decided in either direction within {MAX_T} replies: "
          f"{', '.join(sorted(pairs - decided)) or 'none'}")
    print(f"\n  wrote {TAB / 'betting_ab.csv'}")
    return T


# ------------------------------------------------------------------------ A4

def _external_frames():
    """The three published anchors plus AnnoMI, as (name, spans) pairs."""
    out = []
    try:
        for name, frame in [
            ("WMT24 MT error spans", anc._load_span_events("mt-eval", anc.MT_LABELS)),
            ("data-to-text error spans", anc._load_span_events("d2t-eval", anc.D2T_LABELS)),
            ("RAGTruth hallucination", anc._load_ragtruth()),
        ]:
            out.append((name, frame[["response_id", "label", "position"]]))
    except Exception as e:                      # noqa: BLE001
        print(f"  (external anchors unavailable: {e})")
    annomi = RAW / "annomi" / "AnnoMI-simple.csv"
    if annomi.exists():
        d = pd.read_csv(annomi)
        t = d[d.interlocutor == "therapist"].dropna(
            subset=["main_therapist_behaviour"]).copy()
        span = d.groupby("transcript_id").utterance_id.max()
        t["position"] = (t.utterance_id /
                         t.transcript_id.map(span).clip(lower=1)).clip(0, 1)
        t["response_id"] = t.transcript_id.astype(str)
        t["label"] = t.main_therapist_behaviour
        out.append(("AnnoMI human counsellors",
                    t[["response_id", "label", "position"]]))
    return out


def a4(E: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("A4 — the same test, unchanged, on corpora from other tasks")
    print("=" * 72)
    print("A test that fires everywhere is worthless. On the anchors the")
    print("published score already calls content-driven, this one should")
    print("decline — and it should still fire on the counselling data.\n")
    rows = []
    d = E[E.model == "Claude"][["response_id", "label", "position"]]
    res, bet = both(d)
    rows.append(dict(corpus="counselling (Claude, 20 codes)",
                     SCRIPT=res["SCRIPT"], z=res["z"],
                     edge=round(bet["edge_mean"], 4),
                     rejected=bet["rejected"], tau=bet["tau"],
                     n_replies=bet["n_replies"]))
    for name, spans in _external_frames():
        try:
            res, bet = both(spans)
        except Exception as e:                  # noqa: BLE001
            print(f"  {name}: skipped ({e})")
            continue
        rows.append(dict(corpus=name, SCRIPT=res["SCRIPT"], z=res["z"],
                         edge=round(bet["edge_mean"], 4),
                         rejected=bet["rejected"], tau=bet["tau"],
                         n_replies=bet["n_replies"]))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "betting_transfer.csv", index=False)
    print(T.to_string(index=False))
    print(f"\n  wrote {TAB / 'betting_transfer.csv'}")
    return T


def main() -> None:
    stage = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    E = load_events()
    cal = None
    if stage in ("A1", "A2", "ALL"):
        cal = a1(E)
    if stage in ("A2", "ALL"):
        a2(E, cal)
    if stage in ("A3", "ALL"):
        a3(E)
    if stage in ("A4", "ALL"):
        a4(E)


if __name__ == "__main__":
    main()
