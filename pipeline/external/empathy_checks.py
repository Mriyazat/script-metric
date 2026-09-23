#!/usr/bin/env python3
"""Three checks against the empathy-tactic literature, on its own released data.

  A  SCRIPT, stickiness (their statistic, their code path) and their empathy rating for
     the 22 systems Zhan et al. evaluate on Lend-an-Ear, with matched-size rescoring
  B  coverage of Gueorguieva et al.'s template regex on real tactic strings and on the
     same strings with tactics re-dealt within each reply
  C  tactic stickiness decomposed against a null that permutes turns across
     conversations, on their data and on the benchmark's multi-turn corpora
  D  position profiles per tactic for the template-discovery figure

Needs the MINT release (get_data mint) and nltk's punkt tokenizer for their sentence
split (a regex splitter is used otherwise). Writes tables/mint_systems.csv,
regex_under_null.csv, stickiness_decomposed.csv, out/derived/tactic_profiles.csv."""
import json
import re
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import DATA_DIR, DERIVED, MINT_DIR, MODELS, TAB

from scriptmetric import metric as sm

MINT = MINT_DIR
N_SHUFFLES, SEED = 200, 0
N_MATCH = 1150          # events per system for the matched-size rescoring (smallest LLM system)
rng = np.random.default_rng(SEED)

try:
    from nltk.tokenize import sent_tokenize
    sent_tokenize("ok.")
except Exception:                                   # fallback splitter, same as theirs in spirit
    def sent_tokenize(t):
        return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


# ------------------------------------------------------------------ A. 22 systems

FAMILY = {"baseline1_vanilla": "vanilla prompting", "baseline2_tactic_prompt": "tactic-list prompt",
          "baseline3_tactic_history": "tactic-history prompt",
          "baseline4_vs_vanilla": "verbalized sampling (vanilla prompt)",
          "baseline5_vs_tactic": "verbalized sampling (tactic prompt)",
          "baseline6_vs_tactic_history": "verbalized sampling (history prompt)",
          "baseline7_psychocounsel": "quality-only RL (PsychoCounsel)",
          "baseline8_r1zerodiv": "quality RL + token diversity (R1-Zero-Div)",
          "ours1_q_dkl": "MINT (quality + KL novelty)", "ours2_q_h": "MINT (quality + entropy)",
          "ours3_q_dkl_h": "MINT (quality + KL + entropy)", "gold": "human gold replies"}


def turn_events(entries, system):
    """One (tactic, position) event per tactic on a sentence; position = sentence
    start offset / reply length, from their own sentence split; rank fallback."""
    rows, n_ok = [], 0
    for k, e in enumerate(entries):
        text = e["model_response"].strip()
        sents = sent_tokenize(text) if text else []
        st = e["sentence_tactics"]
        rid = f"{e['conversation_id']}|{k}"
        if len(sents) == len(st):
            n_ok += 1
            pos, cur = [], 0
            for s in sents:
                p = text.find(s, cur)
                pos.append(p / len(text) if p >= 0 else None)
                if p >= 0:
                    cur = p + len(s)
        else:
            pos = [None] * len(st)
        for i, tacs in enumerate(st):
            x = pos[i] if pos[i] is not None else i / max(len(st), 1)
            for t in tacs:
                rows.append((system, rid, t, x))
    return pd.DataFrame(rows, columns=["system", "response_id", "label", "position"]), n_ok


TACTICS10 = ["information", "assistance", "advice", "validation", "emotional_expression", "paraphrasing",
             "self_disclosure", "questioning", "reappraisal", "empowerment"]


def _by_conv(entries):
    out = {}
    for e in entries:
        out.setdefault(e["conversation_id"], []).append(e)
    for c in out:
        out[c].sort(key=lambda e: len(e.get("conversation_history", [])))
    return out


def _tset(e):
    return {t for t, c in e.get("tactic_counts", {}).items() if t in TACTICS10 and c > 0}


def stickiness_zhan(gold, method):
    """Zhan et al.'s evaluation statistic, verbatim from their verify_paper_numbers.py:
    P(tactic in the method's turn t | tactic in the GOLD turn t-1), averaged over
    the ten tactics; the model answers the gold history, so the previous turn is gold."""
    g, m = _by_conv(gold), _by_conv(method)
    hits = {t: [] for t in TACTICS10}
    for cid in sorted(g):
        if cid not in m:
            continue
        for i in range(1, min(len(g[cid]), len(m[cid]))):
            prev, cur = _tset(g[cid][i - 1]), _tset(m[cid][i])
            for t in TACTICS10:
                if t in prev:
                    hits[t].append(int(t in cur))
    probs = [np.mean(v) for v in hits.values() if v]
    return float(np.mean(probs)), sum(len(v) for v in hits.values())


def _stick(pairs):
    tacs = sorted({t for a, b in pairs for t in a | b})
    vals = []
    for t in tacs:
        prev = [(t in b) for a, b in pairs if t in a]
        if len(prev) >= 5:
            vals.append(np.mean(prev))
    return float(np.mean(vals)) if vals else np.nan


def systems_check():
    root = MINT / "evaluation" / "outputs"
    gold = json.load(open(root / "gold" / "conversations_tagged.json"))
    rows = []
    for d in sorted(root.iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        entries = json.load(open(f))
        key = re.sub(r"_Qwen3-(1\.7|4)B$", "", d.name)
        size = re.search(r"Qwen3-(1\.7B|4B)", d.name)
        E, n_ok = turn_events(entries, d.name)
        E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
        res, _ = sm.compute(E[["response_id", "label", "position"]], n_shuffles=N_SHUFFLES, seed=SEED)
        stick, n_pairs = stickiness_zhan(gold, entries)
        # small-sample control: rescore on turn subsamples holding ~N_MATCH events, and extrapolate
        rids = E.response_id.unique()
        per_turn = res["n_events"] / res["n_responses"]
        k = int(min(len(rids), max(30, round(N_MATCH / per_turn))))
        sub = []
        for d_ in range(20):
            ids = np.random.default_rng(d_).choice(rids, k, replace=False)
            r_, _ = sm.compute(E[E.response_id.isin(ids)][["response_id", "label", "position"]], n_shuffles=60, seed=d_)
            sub.append(r_["SCRIPT"])
        ext = sm.extrapolate(E[["response_id", "label", "position"]], n_shuffles=40, draws=6, seed=SEED)
        # expert-protocol empathy rating (their turn-level judge, gpt-oss-120b)
        rf = MINT / "evaluation" / "eval_outputs" / d.name / "gpt-oss-120b_ratings.csv"
        emp = pd.read_csv(rf).aggregated_empathy.mean() if rf.exists() else np.nan
        rows.append(dict(system=d.name, family=FAMILY.get(key, key), size=size.group(1) if size else "human",
                         SCRIPT=res["SCRIPT"], z=res["z"], C=res["C_excess"], M=res["M_excess"],
                         SCRIPT_matched=round(float(np.mean(sub)), 4), SCRIPT_matched_sd=round(float(np.std(sub)), 4),
                         n_turns_matched=k, SCRIPT_extrap=ext["SCRIPT_extrapolated"],
                         stickiness=round(stick, 3), n_pairs=n_pairs, empathy=round(float(emp), 3),
                         n_events=res["n_events"], n_turns=res["n_responses"],
                         events_per_turn=round(res["n_events"] / res["n_responses"], 2),
                         sentence_split_ok=round(n_ok / len(entries), 3)))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "mint_systems.csv", index=False)
    return R


# ------------------------------------------------------- B. their regex under null

LETTER = {"emotional_expression": "X", "validation": "V", "paraphrasing": "P", "advice": "A",
          "information": "I", "empowerment": "E", "reappraisal": "R", "questioning": "Q",
          "assistance": "S", "self_disclosure": "D"}
BASE = "^X?[PV]+[XE]?[AIP]+"
PATTERNS = {1: BASE, 2: BASE + "[VXER]+", 3: BASE + "[VXER]+[AIP]+",
            4: BASE + "[VXER]+[AIP]+[VXER]+", 5: BASE + "[VXER]+[AIP]+[VXER]+[AIP]+"}


def collapse(s):
    return re.sub(r"(.)\1+", r"\1", s)


def match_level(s, k):
    """Their test for pattern k: pattern k matches, or a shorter pattern j<k
    matches and ends the string. Returns matched span length (0 = no match)."""
    m = re.match(PATTERNS[k], s)
    if m:
        return len(m.group(0))
    for j in range(k - 1, 0, -1):
        m = re.match(PATTERNS[j] + "$", s)
        if m:
            return len(m.group(0))
    return 0


def study1_strings():
    d = pd.read_csv(MINT / "data" / "tagger_annotations" / "all_tagged_sentences.csv")
    tac = [t for t in LETTER if t in d.columns]
    d["rid"] = (d.dataset.astype(str) + "|" + d.postID.astype(str) + "|" + d.response_writer.astype(str)
                + "|" + d.whole_response.astype(str).str[:40])
    out = []
    for (rid, w), g in d.groupby(["rid", "response_writer"], sort=False):
        g = g.sort_values("sentenceRANK")
        seq = []
        for _, r in g.iterrows():
            for t in tac:                       # fixed within-sentence order (column order)
                if r[t] == 1:
                    seq.append(LETTER[t])
        if seq:
            out.append((w, "".join(seq)))
    return out


def regex_check(n_null=200):
    strings = study1_strings()
    rows = []
    for w in ["gpt4-turbo", "llama3-70b", "human"]:
        seqs = [s for ww, s in strings if ww == w]
        for k in (1, 3, 5):
            obs = [match_level(collapse(s), k) for s in seqs]
            across = np.mean([m > 0 for m in obs])
            within = np.mean([m / len(collapse(s)) for m, s in zip(obs, seqs) if m > 0]) if any(obs) else 0
            null_across, null_within = [], []
            for _ in range(n_null):
                sh = ["".join(rng.permutation(list(s))) for s in seqs]
                mm = [match_level(collapse(s), k) for s in sh]
                null_across.append(np.mean([m > 0 for m in mm]))
                null_within.append(np.mean([m / len(collapse(s)) for m, s in zip(mm, sh) if m > 0]) if any(mm) else 0)
            rows.append(dict(writer=w, pattern=k, n_replies=len(seqs),
                             across_real=round(100 * across, 1), across_null=round(100 * np.mean(null_across), 1),
                             across_null_sd=round(100 * np.std(null_across), 1),
                             within_real=round(100 * within, 1), within_null=round(100 * np.mean(null_within), 1)))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "regex_under_null.csv", index=False)
    return R


# --------------------------------------------------- C. stickiness decomposed

GROUP = {**{c: "emp_acc" for c in ["VAC", "NAC", "ASAC", "SAC"]},
         **{c: "emp_in" for c in ["VIN", "NIN", "ASIN", "SIN"]},
         **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
         **{c: "quest" for c in ["QOP", "QCL", "TEN"]}}


def _stick_with_null(turn_sets, n_null=300):
    """turn_sets: dict conv -> list of (turn_index, set of labels), one speaker.
    Null: permute, at each turn index, which conversation a turn belongs to."""
    pairs = []
    for turns in turn_sets.values():
        turns.sort()
        pairs += list(zip([s for _, s in turns[:-1]], [s for _, s in turns[1:]]))
    obs = _stick(pairs)
    by_idx = {}
    for cid, turns in turn_sets.items():
        for ti, s in turns:
            by_idx.setdefault(ti, []).append((cid, s))
    null = []
    for _ in range(n_null):
        shuffled = {}
        for ti, lst in by_idx.items():
            cids = [c for c, _ in lst]
            sets = [s for _, s in lst]
            for c, s in zip(cids, rng.permutation(len(sets))):
                shuffled.setdefault(c, []).append((ti, sets[s]))
        p2 = []
        for turns in shuffled.values():
            turns.sort()
            p2 += list(zip([s for _, s in turns[:-1]], [s for _, s in turns[1:]]))
        null.append(_stick(p2))
    null = np.array(null)
    return dict(stickiness=round(obs, 3), null=round(float(null.mean()), 3),
                excess=round(obs - null.mean(), 3), z=round((obs - null.mean()) / null.std(), 2) if null.std() > 0 else np.nan,
                n_pairs=len(pairs))


def stickiness_ours():
    meta = []
    for corpus in ["carebench", "hope"]:
        d = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        d.columns = [c.lstrip("\ufeff").strip() for c in d.columns]
        for i, r in d.iterrows():
            meta.append(dict(corpus=corpus, row=i, conv=f"{corpus}|{r['Conversation']}", turn=int(r["Turn"])))
    meta = pd.DataFrame(meta)
    rows = []
    for layer, f, spk in [("LLM annotator", "llm_span_events.csv", MODELS + ["Human"]),
                          ("clinicians", "span_events.csv", MODELS)]:
        E = pd.read_csv(DERIVED / f).merge(meta, on=["corpus", "row"])
        E["g"] = E.label.map(GROUP).fillna("other")
        for sp in spk:
            for alpha, key in [("5-group", "g"), ("20-code", "label")]:
                ts = {}
                for (conv, turn), g in E[E.model == sp].groupby(["conv", "turn"]):
                    ts.setdefault(conv, []).append((turn, set(g[key])))
                r = _stick_with_null(ts)
                rows.append(dict(data="Cognitive Atrophy Bench (multi-turn)", layer=layer, speaker=sp, alphabet=alpha, **r))
    # their 322 conversations, per model, their 10 tactics
    c = json.load(open(MINT / "data" / "training" / "conversations_322_tagged.json"))
    name = {"gpt-3.5-turbo": "GPT-3.5-turbo", "gpt-4": "GPT-4", "GPT4": "GPT-4", "GPT4-empathy": "GPT-4", "Llama2-70b": "Llama-2-70B"}
    per = {}
    for conv in c:
        m = name.get(conv["model"])
        if m is None:
            continue
        k = 0
        for t in conv["conversation"]:
            if t["role"] == "supporter" and "sentence_tactics" in t:
                k += 1
                per.setdefault(m, {}).setdefault(conv["conversation_id"], []).append(
                    (k, set(x for s in t["sentence_tactics"] for x in s["tactics"])))
    for m, ts in per.items():
        r = _stick_with_null(ts)
        rows.append(dict(data="Zhan et al. conversations", layer="their taggers", speaker=m, alphabet="10 tactics", **r))
    # Lend-an-Ear human supporters (gold)
    gold = json.load(open(MINT / "evaluation" / "outputs" / "gold" / "conversations_tagged.json"))
    ts = {}
    for e in gold:
        ts.setdefault(e["conversation_id"], []).append((len(e["conversation_history"]),
                                                       set(x for s in e["sentence_tactics"] for x in s)))
    r = _stick_with_null(ts)
    rows.append(dict(data="Lend-an-Ear", layer="their taggers", speaker="Human supporters", alphabet="10 tactics", **r))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "stickiness_decomposed.csv", index=False)
    return R


# ------------------------------------------------ D. profile data for the figure

def tactic_profiles():
    E = pd.read_csv(DERIVED / "empathy_tactic_events.csv")
    E["bin"] = np.minimum((E.position * 10).astype(int), 9)
    P = (E.groupby(["layer", "system", "label", "bin"]).size().rename("n").reset_index())
    P.to_csv(DERIVED / "tactic_profiles.csv", index=False)


def main():
    pd.set_option("display.width", 250)
    if not MINT.exists():
        sys.exit(f"MINT release not found at {MINT}; run get_data mint")
    print("== A. 22 systems on Lend-an-Ear")
    R = systems_check()
    print(R.drop(columns=["system"]).to_string(index=False))
    S = R[R["size"] != "human"].dropna(subset=["empathy"])
    for col in ["SCRIPT", "SCRIPT_matched", "SCRIPT_extrap"]:
        print(f"Spearman {col} vs empathy across {len(S)} LLM systems:", round(S[col].rank().corr(S.empathy.rank()), 2),
              f"| {col} vs stickiness:", round(S[col].rank().corr(S.stickiness.rank()), 2),
              f"| {col} vs events/turn:", round(S[col].rank().corr(S.events_per_turn.rank()), 2))
    print("\n== B. their regex under our null")
    print(regex_check().to_string(index=False))
    print("\n== C. stickiness decomposed")
    print(stickiness_ours().to_string(index=False))
    tactic_profiles()
    print("\nwrote mint_systems.csv, regex_under_null.csv, stickiness_decomposed.csv, tactic_profiles.csv")


if __name__ == "__main__":
    main()
