#!/usr/bin/env python3
"""Matched ceilings and small-sample extrapolation — the two portability
add-ons of script_metric.py, exercised on everything this repo measures.

A. Matched ceiling. Raw SCRIPT values are only comparable within one
   annotation scheme, because the alphabet and the event density set how
   high the score can possibly go. The matched ceiling re-arranges each
   corpus's own events into their most scripted order (same responses, same
   positions, same per-response label multisets) and scores that. SCRIPT /
   ceiling — the fraction of achievable rigidity — is a [0,1] quantity
   readable on one scale across schemes.

B. Extrapolation. SCRIPT is downward-biased below ~1,000 events. The
   extrapolated estimator subsamples the corpus, fits SCRIPT against
   1/n_events, and reports the asymptote. This stage validates it against
   known ground truth: subsample the benchmark to N responses, extrapolate,
   compare with the full-sample score.

Writes tables/matched_ceiling.csv and tables/extrapolation_validation.csv.
"""
import importlib.util

import numpy as np
import pandas as pd

from paths import EVENTS, MODELS, PIPELINE, RAW, TAB

import script_metric as sm


def _external_frames():
    """WMT24 / data-to-text / RAGTruth / AnnoMI as (name, spans) pairs."""
    out = []
    spec = importlib.util.spec_from_file_location(
        "anchors", str(PIPELINE / "11_external_anchors.py"))
    anc = importlib.util.module_from_spec(spec)
    anc.__name__ = "anchors"
    spec.loader.exec_module(anc)
    out.append(("WMT24 MT error spans",
                anc._load_span_events("mt-eval", anc.MT_LABELS)))
    out.append(("data-to-text error spans",
                anc._load_span_events("d2t-eval", anc.D2T_LABELS)))
    out.append(("RAGTruth hallucination", anc._load_ragtruth()))
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


def part_a_ceilings(E: pd.DataFrame) -> None:
    print("=" * 72)
    print("A — matched ceilings: fraction of achievable rigidity")
    print("=" * 72)
    rows = []

    def add(name, frame):
        res, _ = sm.compute(frame)
        ceil = sm.matched_ceiling(frame)
        c = max(ceil["SCRIPT"], res["SCRIPT"])
        rows.append(dict(corpus=name, SCRIPT=res["SCRIPT"], z=res["z"],
                         ceiling=round(c, 4),
                         fraction=round(res["SCRIPT"] / c, 4) if c > 0
                         else np.nan,
                         n_events=res["n_events"], n_labels=res["n_labels"]))
        print(f"  {name:42s} SCRIPT {res['SCRIPT']:+.4f}  "
              f"ceiling {c:.4f}  fraction {rows[-1]['fraction']:.3f}")

    for m in MODELS:
        add(f"counselling ({m}, 20 codes)",
            E[E.model == m][["response_id", "label", "position"]])
    for name, frame in _external_frames():
        add(name, frame[["response_id", "label", "position"]])

    T = pd.DataFrame(rows)
    T.to_csv(TAB / "matched_ceiling.csv", index=False)
    print(f"\n  wrote {TAB / 'matched_ceiling.csv'}")


def part_b_extrapolation(E: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("B — extrapolated estimator vs known full-sample truth")
    print("=" * 72)
    rng = np.random.default_rng(42)
    rows = []
    for m in ["Claude", "Llama"]:          # strongest- and weakest-sampled
        d = E[E.model == m][["response_id", "label", "position"]]
        truth, _ = sm.compute(d)
        rids = d.response_id.unique()
        for N in (100, 200, 400):
            for draw in range(5):
                ids = rng.choice(rids, N, replace=False)
                sub = d[d.response_id.isin(ids)]
                naive, _ = sm.compute(sub, n_shuffles=100, seed=draw)
                ext = sm.extrapolate(sub, seed=draw)
                rows.append(dict(model=m, N=N, draw=draw,
                                 naive=naive["SCRIPT"],
                                 extrapolated=ext["SCRIPT_extrapolated"],
                                 se=ext["se"], truth=truth["SCRIPT"]))
        T = pd.DataFrame(rows)
        for N in (100, 200, 400):
            g = T[(T.model == m) & (T.N == N)]
            print(f"  {m:7s} N={N:3d}: naive {g.naive.mean():+.4f}  "
                  f"extrapolated {g.extrapolated.mean():+.4f} "
                  f"(±{g.extrapolated.std():.4f})  truth {truth['SCRIPT']:+.4f}")
    pd.DataFrame(rows).to_csv(TAB / "extrapolation_validation.csv", index=False)
    print(f"\n  wrote {TAB / 'extrapolation_validation.csv'}")


def main() -> None:
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
    part_a_ceilings(E)
    part_b_extrapolation(E)


if __name__ == "__main__":
    main()
