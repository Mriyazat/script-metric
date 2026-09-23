#!/usr/bin/env python3
"""The human anchor: models and therapist on one shared instrument.

Clinician spans exist for the model replies only, so a transparent sentence-level
rule annotator with three labels (questions / empathy / advice) is run over the five
model replies and the therapist's reply to the identical prompts. Its bias cancels in
the comparison; the numbers are comparable only within this table.
Writes tables/therapist_baseline.csv and therapist_js_distance.csv (copies in derived/)."""
import re
import unicodedata

import pandas as pd

from pipeline.common.paths import (CORPORA, DATA_DIR, DERIVED, MODELS, TAB, THERAPIST_COL)

from scriptmetric import metric as sm

SPEAKERS = MODELS + ["Therapist"]
N_BINS = 10
N_SHUFFLES = 200
SEED = 0
MIN_CHARS = 20

# Sentence-level surface patterns. Deliberately simple and inspectable: a
# sentence is a question if it has a '?', otherwise empathy / advice / hedge
# by first matching family. Order matters and is fixed for both sides.
HEDGE = r"\b(might|may|perhaps|could|possibly|it seems|sounds like it may|i wonder)\b"
ADVICE = (r"\b(you (?:should|could|can|need to|must|might want)|try(?:ing)? to|"
          r"consider|start by|focus on|make sure|it(?:'s| is| might be| may be| can be) "
          r"(?:helpful|important|worth)|here are|practice|remember to|reach out|seek|"
          r"schedule|take (?:a|some|time)|set (?:a|boundaries)|talk to|write down|"
          r"allow yourself|give yourself)\b")
EMPATHY = (r"\b(it sounds like|sounds (?:incredibly|really|very|so)|i hear|"
           r"i(?:'m| am) (?:so )?sorry|understandable|your feelings? (?:are|is)|valid|"
           r"makes (?:so much |complete |perfect |total )?sense|that must (?:be|have)|"
           r"you(?:'re| are) not alone|not your fault|takes (?:courage|strength)|"
           r"resilience|thank you for (?:sharing|trusting)|i can (?:only )?imagine|"
           r"it(?:'s| is) (?:completely |perfectly |totally )?"
           r"(?:natural|normal|okay|ok))\b")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("—", "--").replace("–", "-")
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    return re.sub(r"\s+", " ", s).strip().lower()


def rule_annotate(text: str) -> list[tuple[float, str]]:
    """-> [(normalised start position, label)] for every labelled sentence."""
    out, cursor = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        start = text.find(sent, cursor)
        if start < 0:
            continue
        cursor = start + len(sent)
        low = sent.lower()
        if "?" in sent:
            label = "questions"
        elif re.search(EMPATHY, low):
            label = "empathy"
        elif re.search(ADVICE, low):
            label = "advice"
        elif re.search(HEDGE, low):
            label = "questions"       # hedging counted with the question family
        else:
            continue
        out.append((start / len(text), label))
    return out


def build_events() -> pd.DataFrame:
    rows = []
    for corpus in CORPORA:
        df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        tcol = THERAPIST_COL[corpus]
        for i, r in df.iterrows():
            for speaker in SPEAKERS:
                raw = r.get(tcol) if speaker == "Therapist" else r.get(f"{speaker} Output")
                if pd.isna(raw) or not str(raw).strip():
                    continue
                text = norm(raw)
                if len(text) < MIN_CHARS:
                    continue
                for position, label in rule_annotate(text):
                    rows.append(dict(speaker=speaker,
                                     response_id=f"{corpus}|{i}|{speaker}",
                                     label=label, position=position))
    return (pd.DataFrame(rows)
            .sort_values(["response_id", "position"]).reset_index(drop=True))


def main() -> None:
    E = build_events()
    print("events per speaker:")
    print(E.groupby("speaker").size().reindex(SPEAKERS).to_string(), "\n")

    rows, profiles = [], {}
    for speaker in SPEAKERS:
        d = (E[E.speaker == speaker][["response_id", "label", "position"]]
             .sort_values(["response_id", "position"]).reset_index(drop=True))
        res, prof = sm.compute(d, n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED)
        profiles[speaker] = prof
        rows.append(dict(speaker=speaker, SCRIPT=res["SCRIPT"], z=res["z"],
                         C=res["C_excess"], M=res["M_excess"],
                         R_raw=res["R_raw"], R_null=res["R_null"],
                         n_events=res["n_events"],
                         n_responses=res["n_responses"],
                         events_per_response=round(
                             res["n_events"] / max(res["n_responses"], 1), 2)))
    T = pd.DataFrame(rows).set_index("speaker")
    T.to_csv(TAB / "therapist_baseline.csv")
    T.to_csv(DERIVED / "therapist_baseline.csv")
    print("== 3-label rule alphabet, same instrument on both sides ==")
    print(T[["SCRIPT", "C", "M", "z", "n_events", "events_per_response"]].to_string())

    least = T.SCRIPT.idxmin()
    print(f"\nleast scripted speaker: {least} "
          f"({T.loc[least, 'SCRIPT']:.4f}) — "
          f"models span {T.loc[MODELS, 'SCRIPT'].min():.3f}"
          f"-{T.loc[MODELS, 'SCRIPT'].max():.3f}")
    if least != "Therapist":
        print("  WARNING: the human anchor is NOT the least scripted speaker; "
              "the paper's reading of this table does not hold on these data.")

    # structural distance between every pair of speakers, same alphabet
    D = pd.DataFrame(index=SPEAKERS, columns=SPEAKERS, dtype=float)
    for a in SPEAKERS:
        for b in SPEAKERS:
            D.loc[a, b] = (0.0 if a == b else
                           round(sm.profile_distance(profiles[a], profiles[b]), 4))
    D.to_csv(TAB / "therapist_js_distance.csv")
    D.to_csv(DERIVED / "therapist_js_distance.csv")
    print("\n== JS structural distance under the shared alphabet ==")
    print(D.to_string())
    closest = D.loc[MODELS, "Therapist"].astype(float).idxmin()
    print(f"\nclosest model to the therapist: {closest} "
          f"({D.loc[closest, 'Therapist']:.4f})")


if __name__ == "__main__":
    main()
