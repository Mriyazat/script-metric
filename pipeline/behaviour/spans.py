#!/usr/bin/env python3
"""The span layer: what the clinicians highlighted, how much, together with what, and where.

Tables (out/tables/behaviour/):
  spans_long.csv                 one row per highlighted span with its located position
  span_coverage.csv              mean % of reply words highlighted under each code, per model
  empathy_precision.csv          accurate / attempted empathy spans, per model
  span_cooccurrence_<model>.csv  phi coefficients between code presences within a reply
  span_cooccurrence_top.csv      strongest pairs per model (empathy sub-type pairs excluded)
  position_by_group.csv, position_by_group_model.csv   mean start position of the four behaviour families
  last_span_group_pct.csv        family of the last span in the reply, % per model
  advice_momentum.csv            P(next span is advice | current is advice) vs. after a non-advice span
  vulnerability_coupling.csv     Spearman of advice-space against the coded user cues
"""
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr

from pipeline.common import benchmark as bm
from pipeline.common.paths import CODES, GROUPS4, MODELS, TAB_B, USER_ATTRS

KEY = ["corpus", "row", "model"]
FAMILIES = ["empathy_accurate", "empathy_inaccurate", "advice", "questions"]
EMP_ACC, EMP_IN = ["VAC", "NAC", "ASAC", "SAC"], ["VIN", "NIN", "ASIN", "SIN"]


def coverage(S):
    """Per reply and code, highlighted words / reply words; unique span texts per code."""
    u = S.drop_duplicates(KEY + ["code", "text"])
    w = u.groupby(KEY + ["code"]).words.sum().unstack(fill_value=0).reindex(columns=CODES, fill_value=0)
    rw = S.groupby(KEY).reply_words.first()
    pct = w.div(rw, axis=0).clip(upper=1).mul(100)
    return pct.groupby("model").mean().loc[MODELS].T.round(3)


def cooccurrence(S):
    pres = (S.groupby(KEY + ["code"]).size().unstack(fill_value=0) > 0).astype(int)
    pres = pres.reindex(columns=CODES, fill_value=0)
    top = []
    for m in MODELS:
        phi = pres.xs(m, level="model").corr()
        phi.round(3).to_csv(TAB_B / f"span_cooccurrence_{m}.csv")
        pairs = [(a, b, phi.loc[a, b]) for a, b in combinations(CODES, 2)
                 if not ({a, b} <= set(EMP_ACC + EMP_IN)) and pd.notna(phi.loc[a, b])]
        for a, b, v in sorted(pairs, key=lambda x: -abs(x[2]))[:6]:
            top.append(dict(model=m, code_a=a, code_b=b, phi=round(v, 3)))
    return pd.DataFrame(top)


def main():
    S = bm.spans_long()
    S.to_csv(TAB_B / "spans_long.csv", index=False)
    print(f"spans: {len(S)}, located verbatim: {S.located.mean():.3%}")

    coverage(S).to_csv(TAB_B / "span_coverage.csv")

    e = S[S.code.isin(EMP_ACC + EMP_IN)].groupby(KEY + ["code"]).size().unstack(fill_value=0)
    att = e.reindex(columns=EMP_ACC + EMP_IN, fill_value=0).sum(axis=1)
    acc = e.reindex(columns=EMP_ACC, fill_value=0).sum(axis=1)
    P = pd.DataFrame(dict(attempted=att, accurate=acc)).groupby("model").sum().loc[MODELS]
    P["precision"] = P.accurate / P.attempted
    P.round(4).to_csv(TAB_B / "empathy_precision.csv")

    cooccurrence(S).to_csv(TAB_B / "span_cooccurrence_top.csv", index=False)

    # positions: located spans of the four families, one event per (reply, family, position)
    L = S[S.located & S.position.notna()].copy()
    L["group"] = L.code.map(GROUPS4)
    L = L.dropna(subset=["group"]).drop_duplicates(KEY + ["group", "position"])
    L.groupby("group").position.agg(["mean", "median", "count"]).round(3).reindex(FAMILIES) \
        .to_csv(TAB_B / "position_by_group.csv")
    L.pivot_table(index="model", columns="group", values="position").loc[MODELS, FAMILIES].round(3) \
        .to_csv(TAB_B / "position_by_group_model.csv")

    last = L.sort_values("position").groupby(KEY).tail(1)
    lp = pd.crosstab(last.model, last.group, normalize="index").mul(100).loc[MODELS, FAMILIES].round(1)
    lp.to_csv(TAB_B / "last_span_group_pct.csv")

    rows = []
    for m in MODELS:
        c = {"aa": 0, "ax": 0, "xa": 0, "xx": 0}
        for _, g in L[L.model == m].groupby(KEY):
            seq = (g.sort_values("position").group == "advice").astype(int).tolist()
            for a, b in zip(seq, seq[1:]):
                c[("a" if a else "x") + ("a" if b else "x")] += 1
        p_aa = c["aa"] / (c["aa"] + c["ax"])
        p_xa = c["xa"] / (c["xa"] + c["xx"])
        rows.append(dict(model=m, p_advice_after_advice=round(p_aa, 3), p_advice_after_other=round(p_xa, 3),
                         momentum_ratio=round(p_aa / p_xa, 2)))
    M = pd.DataFrame(rows).set_index("model")
    M.to_csv(TAB_B / "advice_momentum.csv")

    # advice-space (advice spans per 100 reply words) against the user cues of the same item
    adv = L[L.group == "advice"].groupby(KEY).size()
    rw = S.groupby(KEY).reply_words.first()
    A = (adv.reindex(rw.index).fillna(0) / rw * 100).rename("advice_space").reset_index()
    A = A.merge(bm.items()[["corpus", "row"] + USER_ATTRS], on=["corpus", "row"])
    rows = []
    for ucol in ["user_sensitivity", "user_evocative", "user_underlying"]:
        d = A.dropna(subset=[ucol])
        rho, p = spearmanr(d[ucol], d.advice_space)
        rows.append(dict(user_cue=ucol, spearman=round(rho, 3), p=p, n=len(d)))
    V = pd.DataFrame(rows)
    V.to_csv(TAB_B / "vulnerability_coupling.csv", index=False)

    print("\nposition by family and model\n", pd.read_csv(TAB_B / "position_by_group_model.csv", index_col=0).to_string())
    print("\nlast span family (%)\n", lp.to_string())
    print("\nadvice momentum\n", M.to_string())
    print("\nadvice-space vs user cue\n", V.to_string(index=False))


if __name__ == "__main__":
    main()
