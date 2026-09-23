#!/usr/bin/env python3
"""Therapist vs. models on the blind LLM layer, where all six speakers answered
the same prompts under one annotator and one 20-code scheme.

Three outputs, all to out/tables/:
  therapist_paired_C.csv       paired prompt-bootstrap of C_excess, model minus therapist
  therapist_llm_ceiling.csv    SCRIPT, C, M, matched ceiling and fraction per speaker
  therapist_density_matched.csv  C_excess within events-per-response bands

Run:  python3 -m pipeline.metric.therapist_paired
"""
import sys
import numpy as np
import pandas as pd

from pipeline.common.paths import DERIVED, TAB, MODELS
from scriptmetric import metric as sm

SPEAKERS = MODELS + ["Human"]
NB = 10


def load():
    d = pd.read_csv(DERIVED / "llm_span_events.csv")
    d["response_id"] = d.corpus + ":" + d.row.astype(str) + ":" + d.model
    d["prompt"] = d.corpus + ":" + d.row.astype(str)
    return d.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)


def frame(d, sp):
    return d[d.model == sp].sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)


def ceilings(d):
    rows = []
    for sp in SPEAKERS:
        f = frame(d, sp)
        res, _ = sm.compute(f, n_shuffles=200)
        ceil = sm.matched_ceiling(f, n_shuffles=50)
        rows.append(dict(speaker=sp, SCRIPT=res["SCRIPT"], C=res["C_excess"], M=res["M_excess"],
                         z=res["z"], ceiling=round(ceil["SCRIPT"], 4),
                         fraction=round(res["SCRIPT"] / ceil["SCRIPT"], 3),
                         n_events=res["n_events"], n_responses=res["n_responses"],
                         events_per_response=round(res["n_events"] / res["n_responses"], 2)))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "therapist_llm_ceiling.csv", index=False)
    print(out.to_string(index=False))


def paired_bootstrap(d, B=300, n_sh=40, seed=0):
    """Resample prompts with replacement; each draw of a prompt is a fresh
    exchangeability block for every speaker, so the therapist and the five
    models are always evaluated on the same prompt set."""
    prompts = sorted(set.intersection(*[set(d[d.model == sp].prompt) for sp in SPEAKERS]))
    dc = d[d.prompt.isin(prompts)]
    labels = sorted(dc.label.astype(str).unique())
    l2i = {l: i for i, l in enumerate(labels)}
    P = {}
    for sp in SPEAKERS:
        f = dc[dc.model == sp].sort_values(["response_id", "position"], kind="stable")
        pos = f.position.to_numpy()
        P[sp] = dict(lab=f.label.astype(str).map(l2i).to_numpy(), pos=pos,
                     xbin=np.minimum((pos * NB).astype(int), NB - 1),
                     rid=f.response_id.to_numpy(),
                     by_prompt={pr: np.where(f.prompt.to_numpy() == pr)[0] for pr in prompts})

    def c_excess(p, idx, rid_key, rng, n_shuffles):
        lab, xbin, pos = p["lab"][idx], p["xbin"][idx], p["pos"][idx]
        rid = pd.factorize(rid_key)[0]
        C = sm._metrics(lab, xbin, rid, len(labels), NB, pos, "exclude")[0]
        null = [sm._metrics(sm._shuffle_within(lab, rid, rng), xbin, rid, len(labels), NB, pos, "exclude")[0]
                for _ in range(n_shuffles)]
        return C - float(np.mean(null))

    point = {sp: c_excess(P[sp], np.arange(len(P[sp]["lab"])), P[sp]["rid"],
                          np.random.default_rng(1), 200) for sp in SPEAKERS}
    rng = np.random.default_rng(seed)
    diffs = {sp: [] for sp in MODELS}
    for b in range(B):
        draw = rng.choice(len(prompts), len(prompts), replace=True)
        cv = {}
        for sp in SPEAKERS:
            p = P[sp]
            blocks = [p["by_prompt"][prompts[j]] for j in draw]
            idx = np.concatenate(blocks)
            copy = np.concatenate([np.full(len(bk), k) for k, bk in enumerate(blocks)])
            rid_key = np.char.add(copy.astype(str), np.char.add("|", p["rid"][idx].astype(str)))
            cv[sp] = c_excess(p, idx, rid_key, rng, n_sh)
        for sp in MODELS:
            diffs[sp].append(cv[sp] - cv["Human"])
        if b % 50 == 0:
            print(f"  bootstrap {b}/{B}", flush=True)
    rows = []
    for sp in MODELS:
        v = np.array(diffs[sp])
        rows.append(dict(model=sp, C_model=round(point[sp], 4), C_therapist=round(point["Human"], 4),
                         delta_C=round(point[sp] - point["Human"], 4),
                         lo95=round(float(np.percentile(v, 2.5)), 4),
                         hi95=round(float(np.percentile(v, 97.5)), 4),
                         share_positive=round(float((v > 0).mean()), 3),
                         n_prompts=len(prompts), B=B))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "therapist_paired_C.csv", index=False)
    print(out.to_string(index=False))


def density_matched(d):
    cnt = d.groupby("response_id").size().rename("n").reset_index()
    d2 = d.merge(cnt, on="response_id")
    rows = []
    for lo, hi in [(3, 4), (5, 7), (8, 10), (11, 999)]:
        for sp in SPEAKERS:
            f = d2[(d2.model == sp) & (d2.n >= lo) & (d2.n <= hi)]
            f = f.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
            n = f.response_id.nunique()
            if n < 30:
                rows.append(dict(band=f"{lo}-{hi}", speaker=sp, C=np.nan, n_responses=n)); continue
            r, _ = sm.compute(f, n_shuffles=100)
            rows.append(dict(band=f"{lo}-{hi}", speaker=sp, C=r["C_excess"], n_responses=n))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "therapist_density_matched.csv", index=False)
    print(out.pivot(index="band", columns="speaker", values="C").round(3).to_string())


if __name__ == "__main__":
    d = load()
    what = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    if what in ("ALL", "CEIL"):
        ceilings(d)
    if what in ("ALL", "PAIRED"):
        paired_bootstrap(d)
    if what in ("ALL", "DENSITY"):
        density_matched(d)
