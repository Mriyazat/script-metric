#!/usr/bin/env python3
"""SCRIPT recomputed inside each annotator's items separately. Writes tables/jury_scripts.csv."""
import pandas as pd

from pipeline.common.paths import EVENTS, MODELS, TAB

from scriptmetric import metric as sm

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model

print("event share by reviewer (%):")
print((E.reviewer.value_counts(normalize=True) * 100).round(1))

rows = []
for rv in ["R1", "R2", "R3", "R4", "R5", "R6"]:
    g = E[E.reviewer == rv]
    for m in MODELS:
        d = g[g.model == m][["response_id", "label", "position"]]
        res, _ = sm.compute(d.sort_values(["response_id", "position"])
                             .reset_index(drop=True),
                            n_bins=10, n_shuffles=200, seed=0)
        rows.append(dict(reviewer=rv, model=m,
                         SCRIPT=round(res["SCRIPT"], 4),
                         z=round(res["z"], 2), n_events=len(d)))

J = pd.DataFrame(rows)
J.to_csv(TAB / "jury_scripts.csv", index=False)
