#!/usr/bin/env python3
"""Leave-one-annotator-out check of the pooled ordering (Claude above the other four).

    python -m pipeline.metric.annotator_tiers R1 ... R6   paired bootstrap without that reviewer (checkpointed)
    python -m pipeline.metric.annotator_tiers COLLECT      tier gap, CI and p per held-out reviewer

Writes tables/loao_tier_gap.csv."""
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import CKPT, EVENTS, MODELS, TAB

from scriptmetric import metric as sm

NB, B_REPS, B_SHUF = 10, 120, 20
CK = CKPT / "loao_ckpt"
CK.mkdir(exist_ok=True)

E = pd.read_csv(EVENTS)
E["unit"] = E.corpus + "|" + E.row.astype(str)
E = E.sort_values(["unit", "model", "position"]).reset_index(drop=True)
REVS = [r for r in E.reviewer.unique() if str(r).startswith("R")]

arg = sys.argv[1]

if arg == "COLLECT":
    rows = []
    for r in sorted(REVS):
        A = np.load(CK / f"{r}.npy")            # (B, 5) SCRIPT per model
        S = {m: A[:, i] for i, m in enumerate(MODELS)}
        # the ordering the pooled Holm-corrected tests resolve: Claude above the other four
        gap = S["Claude"] - (S["Qwen"] + S["Llama"] + S["GPT"] + S["Gemini"]) / 4
        others = np.stack([S[m] for m in MODELS if m != "Claude"])
        wt = S["Claude"] - others.max(0)          # Claude minus the runner-up, per replicate
        p = 2 * min((gap <= 0).mean(), (gap >= 0).mean())
        p = max(p, 1 / len(gap))
        rows.append(dict(held_out=r, n_items=int(E[E.reviewer != r].unit.nunique()),
                         tier_gap=round(gap.mean(), 4),
                         lo=round(np.percentile(gap, 2.5), 4),
                         hi=round(np.percentile(gap, 97.5), 4),
                         p_two_sided=round(p, 4),
                         claude_minus_next_best=round(wt.mean(), 4)))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "loao_tier_gap.csv", index=False)
    print(T)
    sys.exit(0)

rev = arg
sub = E[E.reviewer != rev]
units = sorted(sub.unit.unique())
pre, lab_maps = {}, {}
for m in MODELS:
    d = sub[sub.model == m]
    labels = sorted(d.label.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab_maps[m] = labels
    g = {}
    for u, dd in d.groupby("unit"):
        dd = dd.sort_values("position", kind="stable")
        g[u] = (dd.label.map(l2i).to_numpy(),
                np.minimum((dd.position.to_numpy() * NB).astype(int), NB - 1),
                dd.position.to_numpy())
    pre[m] = g

# Deterministic per-annotator seed (stable across Python runs, unlike hash()).
seed0 = sum(ord(ch) for ch in rev) * 100

out = np.zeros((B_REPS, 5))
for b in range(B_REPS):
    rr = np.random.default_rng(seed0 + b)
    samp = rr.choice(units, size=len(units), replace=True)
    for mi, m in enumerate(MODELS):
        g = pre[m]
        labs, xbs, rids, poss = [], [], [], []
        rid = 0
        for u in samp:
            arr = g.get(u)
            if arr is None or len(arr[0]) == 0:
                continue
            labs.append(arr[0])
            xbs.append(arr[1])
            poss.append(arr[2])
            rids.append(np.full(len(arr[0]), rid))
            rid += 1
        lab = np.concatenate(labs)
        xb = np.concatenate(xbs)
        ri = np.concatenate(rids)
        ps = np.concatenate(poss)
        # slot convention, as in scriptmetric.metric.compute
        C, M, R = sm._metrics(lab, xb, ri, len(lab_maps[m]), NB, pos=ps, ties="exclude")
        null = np.array([sm._metrics(sm._shuffle_within(lab, ri, rr), xb, ri,
                                     len(lab_maps[m]), NB, pos=ps, ties="exclude")[2]
                         for _ in range(B_SHUF)])
        out[b, mi] = R - null.mean()
np.save(CK / f"{rev}.npy", out)
print(rev, "done")
