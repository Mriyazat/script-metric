#!/usr/bin/env python3
"""The momentum term under the two conventions for co-located spans.

Every system on the clinician layer and on the blind LLM layer is scored twice
with identical data, bins, shuffles and seed: once with the slot convention
(``ties="exclude"``, the estimator of the paper: co-located spans form one
multi-label slot and no transition) and once with the legacy chain convention
(``ties="order"``: co-located spans are chained in the stable order of the
input). C is identical under both by construction; the table shows how much of
each system's M was co-annotation.

    python -m pipeline.metric.tie_robustness

Writes out/tables/tie_robustness.csv."""
import pandas as pd

from pipeline.common.paths import DERIVED, EVENTS, MODELS, TAB

from scriptmetric import metric as sm

N_BINS, N_SHUFFLES, SEED = 10, 200, 0


def score(df: pd.DataFrame, layer: str, system: str) -> dict:
    d = (df[["response_id", "label", "position"]]
         .sort_values(["response_id", "position"], kind="stable").reset_index(drop=True))
    slot, _ = sm.compute(d, N_BINS, N_SHUFFLES, SEED, ties="exclude")
    chain, _ = sm.compute(d, N_BINS, N_SHUFFLES, SEED, ties="order")
    row = dict(layer=layer, system=system, events=chain["n_events"],
               transitions_chain=chain["n_transitions"],
               transitions_slot=slot["n_transitions"],
               tied_share=round(chain["n_tied_transitions"] / chain["n_transitions"], 3),
               C=slot["C_excess"],
               SCRIPT_slot=slot["SCRIPT"], M_slot=slot["M_excess"], z_slot=slot["z"],
               SCRIPT_chain=chain["SCRIPT"], M_chain=chain["M_excess"], z_chain=chain["z"])
    print(row, flush=True)
    return row


def main() -> None:
    rows = []
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus.astype(str) + "|" + E.row.astype(str) + "|" + E.model.astype(str)
    for m in MODELS:
        rows.append(score(E[E.model == m], "clinician", m))
    llm = DERIVED / "llm_span_events.csv"
    if llm.exists():
        L = pd.read_csv(llm)
        L["response_id"] = L.corpus.astype(str) + "|" + L.row.astype(str) + "|" + L.model.astype(str)
        for m in MODELS + ["Human"]:
            if (L.model == m).any():
                rows.append(score(L[L.model == m], "llm", m))
    else:
        print(f"{llm} not found; LLM layer skipped")
    pd.DataFrame(rows).to_csv(TAB / "tie_robustness.csv", index=False)
    print(f"wrote {TAB / 'tie_robustness.csv'}")


if __name__ == "__main__":
    main()
