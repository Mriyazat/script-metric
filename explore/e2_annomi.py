#!/usr/bin/env python3
"""SCRIPT on AnnoMI motivational interviewing (exploratory — not in the paper).

AnnoMI is the strongest available external check on the metric, because
everything about it is independent of this project: the conversations were
collected elsewhere, the behaviour codes come from the established MI coding
tradition rather than our manual, the speakers are human counsellors, and the
sessions carry an expert high/low MI-quality label.

    https://github.com/uccollab/AnnoMI

Four questions:

  Q1  Is behavioural scripting present in *human* motivational interviewing?
  Q2  Are LLMs more scripted than high-quality human counsellors?
  Q3  Is low-quality MI more scripted than high-quality MI?
  Q4  Does SCRIPT behave sensibly under an established clinical coding system?

UNIT OF ANALYSIS. In the benchmark a "response" is one reply and position is a
character offset inside it. AnnoMI has no span layer, so the closest analogue
one level up is: a session is the unit, and a therapist utterance's position is
its normalised place in the session timeline. So this measures *session*
choreography — does a counsellor run a fixed arc across the hour — not
within-reply choreography. Comparisons below are therefore made against the
benchmark re-scored at the same unit, never against the paper's headline
number.

    python e2_annomi.py [--shuffles 200]
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from paths import DATA_DIR, EVENTS, MODELS, OUT, RAW   # noqa: E402

import script_metric as sm   # noqa: E402

ANNOMI_URL = "https://github.com/uccollab/AnnoMI/archive/refs/heads/main.zip"
ANNOMI_DIR = RAW / "annomi"
OUTDIR = OUT / "explore"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_BINS = 10
MIN_EVENTS = 150

# The four MI therapist behaviours, and the mapping used to put the benchmark's
# 20 clinician codes on the same alphabet. This is an approximation and the
# main thing to argue about in this file: MI "reflection" is reflecting the
# client's own content back, which the benchmark's empathy/validation codes
# approximate but do not equal; MI "therapist_input" is giving information or
# advice, which the advice codes match closely.
MI_LABELS = ["question", "reflection", "therapist_input", "other"]
CODE_TO_MI = {
    **{c: "question" for c in ["QOP", "QCL"]},
    **{c: "reflection" for c in ["VAC", "NAC", "ASAC", "SAC",
                                 "VIN", "NIN", "ASIN", "SIN"]},
    **{c: "therapist_input" for c in ["DIR", "FIX", "RECT"]},
    **{c: "other" for c in ["SEN", "AUR", "TEN", "TSH", "LMT", "MEN", "INC"]},
}


def fetch() -> Path:
    csv = ANNOMI_DIR / "AnnoMI-simple.csv"
    if csv.exists():
        return csv
    print(f"downloading AnnoMI from {ANNOMI_URL}")
    ANNOMI_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ANNOMI_URL, timeout=180) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    for member in zf.namelist():
        if member.endswith(("AnnoMI-simple.csv", "AnnoMI-full.csv")):
            (ANNOMI_DIR / Path(member).name).write_bytes(zf.read(member))
    return csv


def score(df: pd.DataFrame, n_shuffles: int, label_col="label", seed=0):
    d = (df[["response_id", label_col, "position"]]
         .rename(columns={label_col: "label"})
         .sort_values(["response_id", "position"]).reset_index(drop=True))
    res, prof = sm.compute(d, n_bins=N_BINS, n_shuffles=n_shuffles, seed=seed)
    return res, prof


# ------------------------------------------------------------------- AnnoMI

def load_annomi() -> pd.DataFrame:
    d = pd.read_csv(fetch())
    t = d[d.interlocutor == "therapist"].copy()
    t = t.dropna(subset=["main_therapist_behaviour"])
    # position = where the utterance sits in the session timeline
    span = d.groupby("transcript_id").utterance_id.max()
    t["position"] = t.utterance_id / t.transcript_id.map(span).clip(lower=1)
    t["position"] = t.position.clip(0, 1)
    t["response_id"] = t.transcript_id.astype(str)
    t["label"] = t.main_therapist_behaviour
    return t.sort_values(["response_id", "position"]).reset_index(drop=True)


# -------------------------------------------------- benchmark, same alphabet

def load_benchmark_sessions() -> pd.DataFrame:
    """The two multi-turn corpora, re-scored at AnnoMI's unit and alphabet.

    A conversation is the unit and a span's position is (turn - 1 + within-turn
    position) / n_turns, i.e. where it falls in the whole conversation — the
    same quantity AnnoMI's utterance index measures.
    """
    E = pd.read_csv(EVENTS)
    frames = []
    for corpus in ("carebench", "hope"):
        raw = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False,
                          usecols=["Conversation", "Turn"])
        e = E[E.corpus == corpus].copy()
        e["conv"] = e.row.map(raw.Conversation.astype(str).to_dict())
        e["turn"] = e.row.map(raw.Turn.astype(int).to_dict())
        n_turns = e.groupby("conv").turn.transform("max").clip(lower=1)
        e["position"] = ((e.turn - 1 + e.position) / n_turns).clip(0, 1)
        e["response_id"] = corpus + "|" + e.conv + "|" + e.model
        frames.append(e)
    B = pd.concat(frames, ignore_index=True)
    B["label"] = B.label.map(CODE_TO_MI)
    return B.dropna(subset=["label"]).sort_values(
        ["response_id", "position"]).reset_index(drop=True)


# -------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--only", choices=["annomi", "benchmark", "both"],
                    default="both")
    args = ap.parse_args()
    rows = []

    if args.only in ("annomi", "both"):
        A = load_annomi()
        print(f"AnnoMI: {len(A)} therapist utterances, "
              f"{A.response_id.nunique()} sessions, "
              f"{A.label.nunique()} behaviour codes\n")

        print("=" * 72)
        print("Q1 / Q3 — human MI, pooled and split by expert quality rating")
        print("=" * 72)
        for name, sub in [("all human MI", A),
                          ("high-quality MI", A[A.mi_quality == "high"]),
                          ("low-quality MI", A[A.mi_quality == "low"])]:
            res, _ = score(sub, args.shuffles)
            rows.append(dict(corpus="AnnoMI", system=name, **res))
            print(f"  {name:18s} SCRIPT={res['SCRIPT']:+.4f}  z={res['z']:6.1f}"
                  f"   C={res['C_excess']:+.4f} M={res['M_excess']:+.4f}"
                  f"   ({res['n_events']} utts, {res['n_responses']} sessions)")

        # Matched-n control. High-quality MI has 10x the data of low-quality,
        # and SCRIPT is biased downward at small n, so the raw gap could be
        # entirely an artefact of the smaller stratum reading low. Subsample
        # high-quality down to the low-quality session count and re-score.
        lo = A[A.mi_quality == "low"]
        hi = A[A.mi_quality == "high"]
        n_sessions = lo.response_id.nunique()
        hi_sessions = hi.response_id.unique()
        rng = np.random.default_rng(0)
        draws = []
        for b in range(40):
            pick = rng.choice(hi_sessions, n_sessions, replace=False)
            r, _ = score(hi[hi.response_id.isin(pick)], 60, seed=b)
            draws.append(r["SCRIPT"])
        lo_res, _ = score(lo, args.shuffles)
        hi_m, hi_sd = float(np.mean(draws)), float(np.std(draws))
        gap = lo_res["SCRIPT"] - hi_m
        print(f"\n  matched to {n_sessions} sessions each:")
        print(f"    high-quality  {hi_m:+.4f} +/- {hi_sd:.4f} (40 subsamples)")
        print(f"    low-quality   {lo_res['SCRIPT']:+.4f}")
        print(f"    gap (low - high) = {gap:+.4f}   "
              f"({gap / hi_sd:+.1f} SD of the high-quality draws)")
        rows.append(dict(corpus="AnnoMI", system="high-quality (matched n)",
                         SCRIPT=round(hi_m, 4), z=np.nan,
                         C_excess=np.nan, M_excess=np.nan, R_raw=np.nan,
                         R_null=np.nan, n_events=np.nan,
                         n_responses=n_sessions, n_labels=4,
                         n_bins=N_BINS, n_shuffles=60))

        print("\n" + "=" * 72)
        print("Q4 — does the metric behave sensibly on this coding system?")
        print("=" * 72)
        rng = np.random.default_rng(0)
        ctrl = A.copy()
        ctrl["label"] = rng.permutation(A.label.to_numpy())
        res, _ = score(ctrl, args.shuffles)
        rows.append(dict(corpus="AnnoMI", system="random-label control", **res))
        print(f"  random-label control  SCRIPT={res['SCRIPT']:+.4f} "
              f"z={res['z']:.1f}   (must sit at 0)")

        # per-session scores: is scripting a property of individual sessions?
        per = []
        for sid, g in A.groupby("response_id"):
            if len(g) < 40:
                continue
            gg = g.copy()
            gg["response_id"] = gg.utterance_id.astype(str)   # utterance-level
            r, _ = score(g.assign(response_id=sid), 60)
            per.append(dict(session=sid, quality=g.mi_quality.iloc[0],
                            SCRIPT=r["SCRIPT"], z=r["z"], n=len(g)))
        P = pd.DataFrame(per)
        if len(P):
            P.to_csv(OUTDIR / "annomi_per_session.csv", index=False)
            print(f"\n  per-session ({len(P)} sessions with >=40 utterances):")
            for q, g in P.groupby("quality"):
                print(f"    {q:5s} median SCRIPT {g.SCRIPT.median():+.4f}   "
                      f"IQR [{g.SCRIPT.quantile(.25):+.4f}, "
                      f"{g.SCRIPT.quantile(.75):+.4f}]   n={len(g)}")

    if args.only in ("benchmark", "both"):
        print("\n" + "=" * 72)
        print("Q2 — LLMs vs human counsellors, same 4-label alphabet, "
              "same unit")
        print("=" * 72)
        print("Benchmark multi-turn conversations, 20 codes folded onto the MI"
              "\nalphabet, position = place in the whole conversation.\n")
        B = load_benchmark_sessions()
        for m in MODELS:
            res, _ = score(B[B.model == m], args.shuffles)
            rows.append(dict(corpus="benchmark(MI alphabet)", system=m, **res))
            print(f"  {m:18s} SCRIPT={res['SCRIPT']:+.4f}  z={res['z']:6.1f}"
                  f"   C={res['C_excess']:+.4f} M={res['M_excess']:+.4f}"
                  f"   ({res['n_events']} events, {res['n_responses']} convs)")

    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "annomi_scores.csv", index=False)
    print(f"\nwrote {OUTDIR / 'annomi_scores.csv'}")


if __name__ == "__main__":
    main()
