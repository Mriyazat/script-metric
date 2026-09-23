#!/usr/bin/env python3
"""Testbed 2 under the surface layer: one label per sentence, so M is estimable.

The released tactic taggers of Zhan et al. mark several tactics on one sentence, which
under the slot convention leaves a turn of 2-7 events with only a handful of transitions
and puts the momentum term outside its operating range. This module applies the same
three-label sentence-level rule annotator that places the human therapist and the five
models on one instrument in Testbed 1 (pipeline/metric/therapist_baseline.py: a sentence
with '?' is a question, otherwise empathy / advice / hedge by first matching family),
keeping sentences that match no family as a fourth label 'other' so that every sentence
is one event, to the 315 turns of every one of the 22
systems and of the human gold replies. One event per sentence, so no co-located events,
and both terms of SCRIPT can be read. The labeller sees one sentence at a time and is
therefore also position-blind.

    python -m pipeline.external.mint_surface_layer

Reads raw/mint-empathy/evaluation/outputs/*/conversations_tagged.json and
out/tables/mint_systems.csv (for stickiness, empathy rating, tagger-layer C).
Writes out/tables/mint_surface_layer.csv."""
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.common.paths import MINT_DIR, TAB
from pipeline.external.empathy_checks import FAMILY
from pipeline.metric.therapist_baseline import ADVICE, EMPATHY, HEDGE, MIN_CHARS, norm

from scriptmetric import metric as sm

N_BINS, N_SHUFFLES, SEED = 10, 200, 0
N_MATCH_TURNS = 200          # matched-size rescoring: turns per draw
N_DRAWS = 20


def rule_annotate_all(text):
    """The surface rules of therapist_baseline.rule_annotate (same families, same
    order of tests), with one difference: a sentence that matches no family is kept
    as the label 'other' instead of being dropped, so that every sentence is one
    event and the reply keeps its full sequence of transitions."""
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
            label = "questions"
        else:
            label = "other"
        out.append((start / len(text), label))
    return out


def surface_events(entries, system):
    rows = []
    for k, e in enumerate(entries):
        text = norm(e["model_response"])
        if len(text) < MIN_CHARS:
            continue
        rid = f"{e['conversation_id']}|{k}"
        for position, label in rule_annotate_all(text):
            rows.append(dict(system=system, response_id=rid, label=label, position=position))
    return pd.DataFrame(rows)


def main():
    root = MINT_DIR / "evaluation" / "outputs"
    tagger = pd.read_csv(TAB / "mint_systems.csv").set_index("system")
    rows, frames = [], []
    for d in sorted(root.iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        entries = json.load(open(f))
        E = surface_events(entries, d.name)
        E = E.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
        frames.append(E)
        res, _ = sm.compute(E[["response_id", "label", "position"]], N_BINS, N_SHUFFLES, SEED)
        # matched-size rescoring: the same number of turns for every system
        rng = np.random.default_rng(SEED)
        rids = E.response_id.unique()
        matched = []
        for _ in range(N_DRAWS):
            keep = set(rng.choice(rids, min(N_MATCH_TURNS, len(rids)), replace=False))
            sub = E[E.response_id.isin(keep)][["response_id", "label", "position"]].reset_index(drop=True)
            matched.append(sm.compute(sub, N_BINS, 60, SEED)[0]["SCRIPT"])
        key = re.sub(r"_Qwen3-(1\.7|4)B$", "", d.name)
        size = re.search(r"Qwen3-(1\.7B|4B)", d.name)
        t = tagger.loc[d.name]
        rows.append(dict(system=d.name, family=FAMILY.get(key, key), size=size.group(1) if size else "human",
                         SCRIPT=res["SCRIPT"], z=res["z"], C=res["C_excess"], M=res["M_excess"],
                         SCRIPT_matched=round(float(np.mean(matched)), 4),
                         SCRIPT_matched_sd=round(float(np.std(matched)), 4),
                         n_events=res["n_events"], n_turns=res["n_responses"],
                         events_per_turn=round(res["n_events"] / max(res["n_responses"], 1), 2),
                         share_question=round(float((E.label == "questions").mean()), 3),
                         share_empathy=round(float((E.label == "empathy").mean()), 3),
                         share_advice=round(float((E.label == "advice").mean()), 3),
                         share_other=round(float((E.label == "other").mean()), 3),
                         tagger_C=t["C"], tagger_SCRIPT=t["SCRIPT"], stickiness=t["stickiness"],
                         empathy=t["empathy"], tagger_events_per_turn=t["events_per_turn"]))
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "mint_surface_layer.csv", index=False)
    pd.concat(frames).to_csv(TAB.parent / "derived" / "mint_surface_events.csv", index=False)

    L = T[T["size"] != "human"]
    g = T[T["size"] == "human"].iloc[0]
    print("\n== Testbed 2 under the surface layer (one label per sentence) ==")
    print(T[["system", "SCRIPT", "z", "C", "M", "SCRIPT_matched", "n_events", "events_per_turn"]].to_string(index=False))
    print(f"\nhuman gold: SCRIPT {g.SCRIPT:+.4f} (z {g.z:.1f}), C {g.C:.4f}, M {g.M:+.4f}; "
          f"systems: SCRIPT {L.SCRIPT.min():.3f}-{L.SCRIPT.max():.3f}, C {L.C.min():.3f}-{L.C.max():.3f}, "
          f"M {L.M.min():+.3f}-{L.M.max():+.3f}; systems with M < 0: {(L.M < 0).sum()}")
    print(f"Spearman with tagger layer over 22 systems: SCRIPT vs tagger C {spearmanr(L.SCRIPT, L.tagger_C)[0]:.2f}, "
          f"C vs tagger C {spearmanr(L.C, L.tagger_C)[0]:.2f}")
    print(f"Spearman with empathy rating: SCRIPT {spearmanr(L.SCRIPT, L.empathy)[0]:.2f}, "
          f"matched {spearmanr(L.SCRIPT_matched, L.empathy)[0]:.2f}, C {spearmanr(L.C, L.empathy)[0]:.2f}")
    print(f"Spearman with events/turn: SCRIPT {spearmanr(L.SCRIPT, L.events_per_turn)[0]:.2f}, "
          f"M {spearmanr(L.M, L.events_per_turn)[0]:.2f}; with stickiness: SCRIPT {spearmanr(L.SCRIPT, L.stickiness)[0]:.2f}")
    for size in ["1.7B", "4B"]:
        v = T[(T.family == "vanilla prompting") & (T["size"] == size)].iloc[0]
        m = T[(T.family == "MINT (quality + KL novelty)") & (T["size"] == size)].iloc[0]
        print(f"{size}: vanilla -> MINT(Q+KL): SCRIPT {v.SCRIPT:.3f} -> {m.SCRIPT:.3f}, C {v.C:.3f} -> {m.C:.3f}, "
              f"M {v.M:+.3f} -> {m.M:+.3f}, stickiness {v.stickiness:.3f} -> {m.stickiness:.3f}")
    print(f"\nwrote {TAB / 'mint_surface_layer.csv'}")


if __name__ == "__main__":
    main()
