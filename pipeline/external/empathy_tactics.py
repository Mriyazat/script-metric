#!/usr/bin/env python3
"""SCRIPT on the empathy-tactic annotations released by Gueorguieva et al. (2026) and
Zhan et al. (2026): their Study-1 human coding (GPT-4-turbo, Llama-3-70B, human
writers), their 322 tagged support conversations (GPT-3.5, GPT-4, Llama-2-70B) and the
Lend-an-Ear human supporters. One (tactic, position) event per tactic on a sentence.
Needs the MINT release (get_data mint). Writes tables/anchor_empathy_tactics.csv and
out/derived/empathy_tactic_events.csv."""
import json
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import MINT_DIR, TAB

from scriptmetric import metric as sm

MINT = MINT_DIR
N_BINS, N_SHUFFLES, SEED = 10, 200, 0

TACTICS = ["information", "assistance", "advice", "validation", "emotional_expression",
           "paraphrasing", "solidarity", "spirituality", "self_disclosure", "questioning",
           "reappraisal", "empowerment", "terms_of_endearment", "contextualizing", "gratitude"]


def _pos(sentence, reply, rank, n_sent):
    """Normalised start offset of the sentence in the reply; rank fallback."""
    s, r = str(sentence).strip(), str(reply)
    p = r.find(s) if s else -1
    if p < 0:
        p = r.lower().find(s.lower())
    if p >= 0 and len(r) > 0:
        return p / len(r)
    return (rank - 1) / max(n_sent, 1)


# ------------------------------------------------------------ layer A: coder E

def study1_events():
    d = pd.read_csv(MINT / "data" / "tagger_annotations" / "all_tagged_sentences.csv")
    tac = [t for t in TACTICS if t in d.columns]
    d["response_id"] = (d.dataset.astype(str) + "|" + d.postID.astype(str) + "|"
                        + d.response_writer.astype(str) + "|"
                        + d.whole_response.astype(str).str[:40])
    n_sent = d.groupby("response_id").sentence.transform("count")
    rows = []
    for (rid, w, reply, sent, rank, ns), lab in zip(
            d[["response_id", "response_writer", "whole_response", "sentence",
               "sentenceRANK", ]].assign(ns=n_sent).itertuples(index=False), d[tac].to_numpy()):
        x = _pos(sent, reply, rank, ns)
        for t, v in zip(tac, lab):
            if v == 1:
                rows.append((w, rid, t, x))
    E = pd.DataFrame(rows, columns=["system", "response_id", "label", "position"])
    E["layer"] = "Study-1 human coding"
    return E


# ------------------------------------------- layer B: 322 tagged conversations

def conversations_events():
    c = json.load(open(MINT / "data" / "training" / "conversations_322_tagged.json"))
    name = {"gpt-3.5-turbo": "GPT-3.5-turbo", "gpt-4": "GPT-4", "GPT4": "GPT-4",
            "GPT4-empathy": "GPT-4", "Llama2-70b": "Llama-2-70B"}
    rows = []
    for conv in c:
        sysname = name.get(conv["model"])
        if sysname is None:                       # 'IC' = unknown source, skipped
            continue
        k = 0
        for turn in conv["conversation"]:
            if turn["role"] != "supporter" or "sentence_tactics" not in turn:
                continue
            k += 1
            rid = f"{conv['conversation_id']}|{k}"
            reply = turn["content"]
            sents = turn["sentence_tactics"]
            for i, s in enumerate(sents, 1):
                x = _pos(s["sentence"], reply, i, len(sents))
                for t in s["tactics"]:
                    rows.append((sysname, rid, t, x))
    E = pd.DataFrame(rows, columns=["system", "response_id", "label", "position"])
    E["layer"] = "Real conversations (tagger)"
    return E


# ------------------------------------------------- layer C: Lend-an-Ear humans

def lend_an_ear_events():
    d = pd.read_csv(MINT / "data" / "lend_an_ear_eval" / "tactic_results" /
                    "sentence_level_tactics.csv")
    tac = [t for t in TACTICS if t in d.columns]
    d["response_id"] = d.conversation_id.astype(str) + "|" + d.supporter_turn_index.astype(str)
    n_sent = d.groupby("response_id").sentence.transform("count")
    rows = []
    for (rid, reply, sent, rank, ns), lab in zip(
            d[["response_id", "whole_response", "sentence", "sentenceRANK"]]
            .assign(ns=n_sent).itertuples(index=False), d[tac].to_numpy()):
        x = _pos(sent, reply, rank, ns)
        for t, v in zip(tac, lab):
            if v == 1:
                rows.append(("Human supporters (Lend-an-Ear)", rid, t, x))
    E = pd.DataFrame(rows, columns=["system", "response_id", "label", "position"])
    E["layer"] = "Real conversations (tagger)"
    return E


# --------------------------------------------------------------------- scoring

def score(E, system, layer, seed=SEED):
    d = E[(E.system == system) & (E.layer == layer)]
    d = d.sort_values(["response_id", "position"]).reset_index(drop=True)
    res, _ = sm.compute(d[["response_id", "label", "position"]], n_bins=N_BINS,
                        n_shuffles=N_SHUFFLES, seed=seed)
    ceil = sm.matched_ceiling(d[["response_id", "label", "position"]], n_bins=N_BINS,
                              n_shuffles=N_SHUFFLES, seed=seed)
    # random-label control: this system's own events, labels re-dealt within reply
    rng = np.random.default_rng(seed)
    ctrl = d.copy()
    ctrl["label"] = ctrl.groupby("response_id")["label"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    cres, _ = sm.compute(ctrl[["response_id", "label", "position"]], n_bins=N_BINS,
                         n_shuffles=N_SHUFFLES, seed=seed + 1)
    return dict(layer=layer, system=system, SCRIPT=res["SCRIPT"], z=res["z"],
                C=res["C_excess"], M=res["M_excess"], ceiling=ceil["SCRIPT"],
                fraction=round(res["SCRIPT"] / ceil["SCRIPT"], 4) if ceil["SCRIPT"] else np.nan,
                control=cres["SCRIPT"], n_events=res["n_events"],
                n_responses=res["n_responses"], n_labels=res["n_labels"],
                events_per_response=round(res["n_events"] / res["n_responses"], 2))


def main():
    if not MINT.exists():
        sys.exit(f"MINT release not found at {MINT}; run get_data mint")
    E = pd.concat([study1_events(), conversations_events(), lend_an_ear_events()],
                  ignore_index=True)
    print(E.groupby(["layer", "system"]).agg(events=("label", "size"),
                                              replies=("response_id", "nunique")))
    order = [("Study-1 human coding", "gpt4-turbo"),
             ("Study-1 human coding", "llama3-70b"),
             ("Study-1 human coding", "human"),
             ("Real conversations (tagger)", "GPT-3.5-turbo"),
             ("Real conversations (tagger)", "GPT-4"),
             ("Real conversations (tagger)", "Llama-2-70B"),
             ("Real conversations (tagger)", "Human supporters (Lend-an-Ear)")]
    rows = [score(E, s, l) for l, s in order]
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "anchor_empathy_tactics.csv", index=False)
    E.to_csv(TAB.parent / "derived" / "empathy_tactic_events.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    print(f"wrote {TAB / 'anchor_empathy_tactics.csv'}")


if __name__ == "__main__":
    main()
