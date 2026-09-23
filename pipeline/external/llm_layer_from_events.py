#!/usr/bin/env python3
"""Rebuild the blind-LLM annotation layer of Testbed 1 from the shipped event file.

The per-reply annotator cache (out/derived/llm_annotator/) contains verbatim quotes
of the benchmark text and is not redistributed. Its reduction to events, one
(corpus, row, model, label, position) row per located span with no text, is
shipped as data/llm_span_events.csv. This script

  1. copies it to out/derived/llm_span_events.csv, where the pipeline expects it, and
  2. recomputes out/tables/llm_annotator_comparison.csv (Table 2, blind-LLM block,
     plus the random-label control) from those events and the clinician-layer scores.

Run:  python3 -m pipeline.external.llm_layer_from_events
"""
import shutil
import numpy as np
import pandas as pd

from pipeline.common.paths import DERIVED, MODELS, REPO, TAB
from scriptmetric import metric as sm

SHIPPED = REPO / "data" / "llm_span_events.csv"
TARGET = DERIVED / "llm_span_events.csv"


def main():
    if not TARGET.exists():
        if not SHIPPED.exists():
            raise SystemExit("no LLM layer: run pipeline.external.llm_annotator with an API key, "
                             "or restore data/llm_span_events.csv")
        shutil.copy(SHIPPED, TARGET)
        print("copied", SHIPPED, "->", TARGET)
    d = pd.read_csv(TARGET)
    d["response_id"] = d.corpus + ":" + d.row.astype(str) + ":" + d.model
    d = d.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)

    clin = pd.read_csv(DERIVED / "validation_results.csv").set_index("system") \
        if (DERIVED / "validation_results.csv").exists() else None
    rows = []
    for sp in MODELS + ["Human"]:
        f = d[d.model == sp][["response_id", "label", "position"]].reset_index(drop=True)
        r, _ = sm.compute(f, n_shuffles=200, seed=0)
        row = dict(model=sp, SCRIPT_llm=r["SCRIPT"], z_llm=r["z"], C_llm=r["C_excess"], M_llm=r["M_excess"],
                   n_events_llm=r["n_events"])
        if clin is not None and sp in clin.index:
            c = clin.loc[sp]
            row.update(SCRIPT_clin=c["SCRIPT"], z_clin=c["z"], C_clin=c["C_excess"], M_clin=c["M_excess"],
                       n_events_clin=c.get("n_events", np.nan))
        rows.append(row)
    # random-label control: the five models' LLM events with labels re-dealt within each response
    rng = np.random.default_rng(0)
    ctl = d[d.model.isin(MODELS)][["response_id", "label", "position"]].copy()
    ctl["label"] = ctl.groupby("response_id")["label"].transform(lambda s: rng.permutation(s.to_numpy()))
    r, _ = sm.compute(ctl.reset_index(drop=True), n_shuffles=200, seed=0)
    rows.append(dict(model="random-label control (LLM events)", SCRIPT_llm=r["SCRIPT"], z_llm=r["z"],
                     C_llm=r["C_excess"], M_llm=r["M_excess"], n_events_llm=r["n_events"]))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "llm_annotator_comparison.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
