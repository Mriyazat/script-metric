#!/usr/bin/env python3
"""How much of profile identification is arrangement, and how much is the behaviour mix?

Held-out-corpus protocol of pipeline.metric.identification (enrol on three corpora,
probe k replies from the fourth, 200 draws per model and held-out corpus), scored
four ways:
  full        position table + transition table (the paper's profile)
  position    position table only
  mix         label frequencies only (no position, no order)
  redealt     full profile, but the probe's labels re-dealt within each reply
              (the paper's own null applied to the probe)

Writes tables/identification_baselines.csv.
Run:  python3 -m pipeline.metric.identification_baselines [--draws 200]
"""
import argparse
import numpy as np
import pandas as pd

from pipeline.common.paths import EVENTS, MODELS, TAB
from pipeline.metric.identification import build_profile, N_BINS

K_VALUES = [1, 5, 20]
ALPHA = 0.5


def tables(ref):
    P = np.array(ref["position"]) + ALPHA
    T = np.array(ref["transition"]) + ALPHA
    logP = np.log(P / P.sum(0, keepdims=True))                 # log P(label | bin)
    logM = np.log(P.sum(1) / P.sum())                          # log P(label)
    logT = np.log(T / T.sum(1, keepdims=True))                 # log P(current | previous)
    return logP, logM, logT


def per_response(pool, ref, mode):
    labels = ref["labels"]; l2i = {l: i for i, l in enumerate(labels)}
    logP, logM, logT = tables(ref)
    backoff = np.log(1.0 / len(labels))
    d = pool
    rid = d.response_id.to_numpy(); lab = np.array([l2i.get(str(x), -1) for x in d.label.to_numpy()])
    posv = d.position.to_numpy(); bins = np.minimum((posv * N_BINS).astype(int), N_BINS - 1)
    totals, counts = {}, {}
    prev, prev_rid, prev_pos = -1, None, None
    for r, li, b, x in zip(rid, lab, bins, posv):
        if r != prev_rid:
            prev, prev_rid, prev_pos = -1, r, None
        if li < 0:
            val = backoff
        elif mode == "mix":
            val = logM[li]
        else:
            val = logP[li, b]
        if mode in ("full", "redealt") and prev >= 0 and li >= 0 and x != prev_pos:
            val += logT[prev, li]
        totals[r] = totals.get(r, 0.0) + val; counts[r] = counts.get(r, 0) + 1
        prev, prev_pos = li, x
    return totals, counts


def redeal(pool, rng):
    """Permute labels within each response, positions fixed."""
    pool = pool.copy()
    pool["label"] = pool.groupby("response_id")["label"].transform(lambda s: rng.permutation(s.to_numpy()))
    return pool


def play(E, enrol_mask, probe_mask, k, draws, mode, seed):
    refs = {m: build_profile(E[enrol_mask & (E.model == m)]) for m in MODELS}
    rng = np.random.default_rng(seed)
    correct = played = 0
    for true_m in MODELS:
        pool = E[probe_mask & (E.model == true_m)].sort_values(["response_id", "position"], kind="stable")
        if mode == "redealt":
            pool = redeal(pool, rng)
        rids = pool.response_id.unique()
        if len(rids) < k:
            continue
        order = {r: i for i, r in enumerate(rids)}
        tot = np.zeros((len(MODELS), len(rids))); cnt = np.zeros(len(rids))
        for mi, m in enumerate(MODELS):
            totals, counts = per_response(pool, refs[m], mode)
            for r, v in totals.items():
                tot[mi, order[r]] = v
            if mi == 0:
                for r, c in counts.items():
                    cnt[order[r]] = c
        for _ in range(draws):
            idx = rng.choice(len(rids), k, replace=False)
            n = cnt[idx].sum()
            scores = tot[:, idx].sum(1) / n
            correct += MODELS[int(np.argmax(scores))] == true_m; played += 1
    return correct, played


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=200); args = ap.parse_args()
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
    corpora = sorted(E.corpus.unique())
    rows = []
    for mode in ["full", "position", "mix", "redealt"]:
        for k in K_VALUES:
            ok = n = 0
            for held in corpora:
                a, b = play(E, E.corpus.ne(held), E.corpus.eq(held), k, args.draws, mode, seed=2000 + k)
                ok += a; n += b
            rows.append(dict(mode=mode, k=k, correct=ok, games=n, accuracy=round(ok / n, 3)))
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "identification_baselines.csv", index=False)
    print(T.pivot(index="mode", columns="k", values="accuracy").to_string())


if __name__ == "__main__":
    main()
