#!/usr/bin/env python3
"""Reduce the clinician span layer to (label, position) events.

Every highlight is located verbatim in its normalised reply; the event keeps the
span code and the normalised start offset. Writes out/derived/span_events.csv."""
import pandas as pd

from pipeline.common.benchmark import load_annotated, norm
from pipeline.common.paths import CODES, CORPORA, EVENTS, MODNUM


def main() -> None:
    rows = []
    n_total = n_found = 0
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m, n in MODNUM.items():
                out = r.get(f"{m} Output")
                if pd.isna(out):
                    continue
                txt = norm(out)
                L = len(txt)
                if L == 0:
                    continue
                seen = set()
                for code in CODES:
                    col = f"Response {n}_{code}"
                    if col not in df.columns:
                        continue
                    v = r[col]
                    if pd.isna(v) or not str(v).strip():
                        continue
                    for sp in str(v).split(" | "):
                        spn = norm(sp)
                        if not spn or spn == "#name?":
                            continue
                        n_total += 1
                        p = txt.find(spn)
                        if p < 0:
                            continue
                        n_found += 1
                        key = (code, p)
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(dict(corpus=corpus, row=i, reviewer=r.get("reviewer"), model=m,
                                         label=code, position=p / L))
    E = pd.DataFrame(rows)
    E.to_csv(EVENTS, index=False)
    print(f"spans searched {n_total}, located {n_found} ({100 * n_found / n_total:.2f}%), "
          f"events after dedupe {len(E)}")
    print(f"wrote {EVENTS}")


if __name__ == "__main__":
    main()
