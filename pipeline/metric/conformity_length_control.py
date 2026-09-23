#!/usr/bin/env python3
"""Positive control for the matched-mix test of pipeline.metric.conformity (Testbed 2).

Within the same matched groups (same system, same label multiset on the surface
layer), the main test asks whether the judge's rating differs between the turn
that conforms more to the system's blueprint and the one that conforms less.
Here we ask, in the same groups, whether the rating differs between the LONGER
and the SHORTER turn (characters of the normalised response). If the judge
responds to length at matched mix but not to arrangement, the null on
arrangement is not a lack of power in the design.

Writes tables/conformity_length_control.csv.
Run:  python3 -m pipeline.metric.conformity_length_control
"""
import json
import numpy as np
import pandas as pd

from pipeline.common.paths import DERIVED, MINT_DIR, TAB
from pipeline.metric.therapist_baseline import norm

SEED, B = 0, 2000


def lengths():
    rows = []
    for d in sorted((MINT_DIR / "evaluation" / "outputs").iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        for k, e in enumerate(json.load(open(f))):
            rows.append(dict(system=d.name, response_id=f"{e['conversation_id']}|{k}",
                             n_chars=len(norm(e["model_response"]))))
    return pd.DataFrame(rows)


def paired(D, value_col, outcome="empathy"):
    """All within-group pairs; diff = outcome of the higher-`value_col` turn minus the lower."""
    rows = []
    for (sysname, key), g in D.groupby(["system", "key"]):
        if len(g) < 2:
            continue
        a = g.to_dict("records")
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                hi, lo = (a[i], a[j]) if a[i][value_col] >= a[j][value_col] else (a[j], a[i])
                if hi[value_col] == lo[value_col]:
                    continue
                rows.append(dict(group=f"{sysname}|{key}", diff=hi[outcome] - lo[outcome],
                                 gap=hi[value_col] - lo[value_col]))
    return pd.DataFrame(rows)


def cluster_ci(P, rng):
    g = P.groupby("group")["diff"].mean()
    boots = [g.sample(frac=1, replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(B)]
    return float(g.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(g), len(P)


def main():
    rng = np.random.default_rng(SEED)
    D = pd.read_csv(DERIVED / "conformity_tb2_turns_own.csv").merge(lengths(), on=["system", "response_id"], how="inner")
    D = D[D.family != "human"] if "human" in set(D.family) else D
    out = []
    for name, col in [("conformity (arrangement)", "excess"), ("length (characters)", "n_chars")]:
        P = paired(D, col)
        m, lo, hi, ng, npairs = cluster_ci(P, rng)
        out.append(dict(contrast=name, groups=ng, pairs=npairs, mean_diff=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        median_gap=round(float(P.gap.median()), 3)))
    # length effect restricted to the half of pairs with the largest length gap
    P = paired(D, "n_chars"); P = P[P.gap >= P.gap.median()]
    m, lo, hi, ng, npairs = cluster_ci(P, rng)
    out.append(dict(contrast="length, top half of the length gap", groups=ng, pairs=npairs, mean_diff=round(m, 4),
                    ci_lo=round(lo, 4), ci_hi=round(hi, 4), median_gap=round(float(P.gap.median()), 3)))
    T = pd.DataFrame(out)
    T.to_csv(TAB / "conformity_length_control.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
