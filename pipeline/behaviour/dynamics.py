#!/usr/bin/env python3
"""Turn dynamics in the two multi-turn corpora.

Tables (out/tables/behaviour/):
  scores_by_turn.csv         mean attribute score per turn bin, per model and pooled
  scores_turn_trend.csv      Spearman correlation of each attribute with the turn index, pooled and per model
  advice_autocorrelation.csv lag-1 Spearman autocorrelation of advice density within conversations, against
                             a null that permutes turn order within each conversation
  turn1_fate.csv             Spearman of turn-1 advice density against the mean of the later turns
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.common import benchmark as bm
from pipeline.common.paths import ATTRS, GROUPS4, MODELS, MULTITURN_CORPORA, TAB_B

KEY = ["corpus", "row", "model"]
TURN_BINS = [0, 2, 4, 6, 8, 100]
TURN_LABELS = ["1-2", "3-4", "5-6", "7-8", "9+"]
N_PERM, SEED = 200, 0


def advice_density():
    """Advice words per 100 reply words, per multi-turn reply."""
    S = pd.read_csv(TAB_B / "spans_long.csv")
    S = S[S.corpus.isin(MULTITURN_CORPORA) & S.located]
    S = S.assign(group=S.code.map(GROUPS4))
    adv = S[S.group == "advice"].drop_duplicates(KEY + ["text"]).groupby(KEY).words.sum()
    R = bm.replies()
    R = R[R.corpus.isin(MULTITURN_CORPORA) & R.model.isin(MODELS) & (R.text.str.len() > 0)]
    rw = R.set_index(KEY).text.str.split().str.len()
    return (adv.reindex(rw.index).fillna(0) / rw * 100).rename("advice_density").reset_index()


def lag1(df, col):
    a, b = [], []
    for _, g in df.sort_values("turn").groupby(["corpus", "conversation"]):
        v = g[col].to_numpy()
        a += list(v[:-1])
        b += list(v[1:])
    return np.array(a), np.array(b)


def main():
    I = bm.items()[["corpus", "row", "conversation", "turn"]]
    I = I[I.corpus.isin(MULTITURN_CORPORA)].assign(conversation=lambda d: d.conversation.astype(str))
    S = bm.scores_long().merge(I, on=["corpus", "row"])
    S["FIX"] = np.nan
    S.loc[S.attribute == "FIX", "value"] = (S.loc[S.attribute == "FIX", "value"] > 0).astype(float)
    S["turn_bin"] = pd.cut(S.turn, TURN_BINS, labels=TURN_LABELS)

    by_turn = S.pivot_table(index=["attribute", "model"], columns="turn_bin", values="value", observed=True)
    pooled = S.pivot_table(index="attribute", columns="turn_bin", values="value", observed=True)
    pooled["model"] = "pooled"
    out = pd.concat([by_turn.reset_index(), pooled.reset_index()], ignore_index=True)
    out.round(3).to_csv(TAB_B / "scores_by_turn.csv", index=False)

    rows = []
    for a in ATTRS:
        d = S[S.attribute == a].dropna(subset=["value"])
        rho, p = spearmanr(d.turn, d.value)
        rows.append(dict(attribute=a, model="pooled", spearman=round(rho, 3), p=p, n=len(d)))
        for m in MODELS:
            dm = d[d.model == m]
            rho, p = spearmanr(dm.turn, dm.value)
            rows.append(dict(attribute=a, model=m, spearman=round(rho, 3), p=p, n=len(dm)))
    T = pd.DataFrame(rows)
    T.to_csv(TAB_B / "scores_turn_trend.csv", index=False)

    A = advice_density().merge(I, on=["corpus", "row"])
    rng = np.random.default_rng(SEED)
    rows = []
    for m in MODELS:
        d = A[A.model == m]
        r_obs, _ = spearmanr(*lag1(d, "advice_density"))
        null = []
        for _ in range(N_PERM):
            ds = d.assign(shuf=d.groupby(["corpus", "conversation"]).advice_density.transform(rng.permutation))
            null.append(spearmanr(*lag1(ds, "shuf"))[0])
        null = np.array(null)
        rows.append(dict(model=m, lag1_autocorr=round(r_obs, 3), null_mean=round(null.mean(), 3),
                         p_vs_null=round((np.sum(null >= r_obs) + 1) / (N_PERM + 1), 4)))
    pd.DataFrame(rows).to_csv(TAB_B / "advice_autocorrelation.csv", index=False)

    rows = []
    t1 = A[A.turn == 1].set_index(["corpus", "conversation", "model"]).advice_density
    later = A[A.turn >= 2].groupby(["corpus", "conversation", "model"]).advice_density.mean()
    J = pd.DataFrame(dict(t1=t1, later=later)).dropna().reset_index()
    for m in MODELS + ["pooled"]:
        d = J if m == "pooled" else J[J.model == m]
        rho, p = spearmanr(d.t1, d.later)
        rows.append(dict(model=m, spearman_turn1_vs_later=round(rho, 3), p=p, n_conversations=len(d)))
    F = pd.DataFrame(rows)
    F.to_csv(TAB_B / "turn1_fate.csv", index=False)

    q = pooled.loc["QOC"]
    print(f"pooled question openness (QOC) by turn: {q[TURN_LABELS[0]]:.2f} -> {q[TURN_LABELS[-1]]:.2f}; "
          f"Spearman vs turn {T[(T.attribute == 'QOC') & (T.model == 'pooled')].spearman.iloc[0]:.3f}")
    print("\nadvice persistence\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
