#!/usr/bin/env python3
"""Matched ceilings and small-sample extrapolation on every corpus in the paper.

The matched ceiling scores the most scripted arrangement of a corpus's own events, so
SCRIPT / ceiling is a fraction of achievable rigidity comparable across schemes. The
extrapolated estimator fits SCRIPT against 1/n over subsamples; it is validated here
against the known full-sample scores. Needs the external anchors downloaded.
Writes tables/matched_ceiling.csv and extrapolation_validation.csv."""
import numpy as np
import pandas as pd

from pipeline.common.paths import EVENTS, MODELS, RAW, TAB
from pipeline.external import anchors as anc

from scriptmetric import metric as sm


def _external_frames():
    """WMT24 / data-to-text / RAGTruth / propaganda / AnnoMI as (name, spans) pairs."""
    out = []
    out.append(("WMT24 MT error spans",
                anc._load_span_events("mt-eval", anc.MT_LABELS)))
    out.append(("data-to-text error spans",
                anc._load_span_events("d2t-eval", anc.D2T_LABELS)))
    out.append(("RAGTruth hallucination", anc._load_ragtruth()))
    out.append(("propaganda techniques (human news)", anc._load_propaganda("human")))
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
                         C=res["C_excess"], M=res["M_excess"],
                         ceiling=round(c, 4),
                         fraction=round(res["SCRIPT"] / c, 4) if c > 0
                         else np.nan,
                         n_events=res["n_events"], n_responses=res["n_responses"],
                         n_labels=res["n_labels"]))
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
