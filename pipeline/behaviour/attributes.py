#!/usr/bin/env python3
"""Attribute scores and global flags: how much of each behaviour each model shows.

Tables (out/tables/behaviour/):
  scores_long.csv            one row per (item, model, attribute)
  scores_by_model.csv        mean score per attribute and model; Fix-It also as share of replies with any solution
  scores_tests.csv           Friedman test across the five models per (corpus, attribute), BH-corrected
  scores_spearman.csv        Spearman correlation among the ten attributes (pooled replies, Fix-It binarised)
  scores_pca.csv             explained variance of the standardised attribute matrix (Fix-It binarised)
  flags_long.csv, flags_by_model.csv (rate with Wilson 95% CI), flags_by_user_cue.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.common import benchmark as bm
from pipeline.common.paths import ATTRS, MODELS, TAB_B


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def main():
    S = bm.scores_long()
    S.to_csv(TAB_B / "scores_long.csv", index=False)
    wide = S.pivot_table(index=["corpus", "row", "model"], columns="attribute", values="value")[ATTRS]

    by_model = wide.groupby("model").mean().loc[MODELS].T
    by_model.loc["FIX_any"] = wide.groupby("model")["FIX"].apply(lambda v: (v > 0).mean()).loc[MODELS]
    by_model.round(3).to_csv(TAB_B / "scores_by_model.csv")

    rows = []
    for (corpus, a), d in S.groupby(["corpus", "attribute"]):
        w = d.pivot_table(index="row", columns="model", values="value").dropna()
        stat, p = stats.friedmanchisquare(*[w[m] for m in MODELS])
        rows.append(dict(corpus=corpus, attribute=a, n_items=len(w), friedman_chi2=stat, p=p))
    T = pd.DataFrame(rows)
    T["q_bh"] = bh(T.p)
    T["significant"] = T.q_bh < 0.05
    T.to_csv(TAB_B / "scores_tests.csv", index=False)

    # Fix-It levels 1/2 code the appropriateness of a solution, not its amount, so
    # wherever the attribute is averaged or correlated it is read as 1[FIX>0].
    X = wide.dropna().copy()
    X["FIX"] = (X["FIX"] > 0).astype(float)
    X.corr(method="spearman").round(3).to_csv(TAB_B / "scores_spearman.csv")
    Z = (X - X.mean()) / X.std(ddof=1)
    ev = np.linalg.svd(Z.to_numpy(), compute_uv=False) ** 2
    ev = ev / ev.sum()
    pd.DataFrame(dict(component=np.arange(1, len(ev) + 1), explained=ev.round(4),
                      cumulative=np.cumsum(ev).round(4))).to_csv(TAB_B / "scores_pca.csv", index=False)

    F = bm.flags_long()
    F.to_csv(TAB_B / "flags_long.csv", index=False)
    rows = []
    for (m, f), d in F.groupby(["model", "flag"]):
        k, n = int(d.value.sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append(dict(model=m, flag=f, n=n, k=k, rate=k / n, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).round(4).to_csv(TAB_B / "flags_by_model.csv", index=False)

    I = bm.items()
    Fm = F.merge(I[["corpus", "row", "user_sensitivity", "user_request_info"]], on=["corpus", "row"])
    cue = pd.concat({u: Fm.groupby(["flag", u]).value.mean().unstack() for u in
                     ["user_sensitivity", "user_request_info"]}, names=["user_cue"])
    cue.round(4).to_csv(TAB_B / "flags_by_user_cue.csv")

    print("mean scores by model\n", by_model.round(2).to_string())
    print(f"\nsignificant model differences: {T.significant.sum()} / {len(T)} (corpus x attribute, BH q<0.05)")
    print(f"PCA explained variance: PC1 {ev[0]:.3f}, five components {np.cumsum(ev)[4]:.3f}")
    print("\nflag rates\n", pd.DataFrame(rows).pivot(index="flag", columns="model", values="rate")[MODELS].round(3).to_string())


if __name__ == "__main__":
    main()
