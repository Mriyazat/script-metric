import re
import unicodedata

import pandas as pd

from paths import CODES, CORPORA, DATA_DIR, EVENTS, MODNUM


def norm(s: str) -> str:
    """Normalise text so annotation spans match the response verbatim."""
    s = unicodedata.normalize("NFKC", str(s)).replace("—", "--").replace("–", "-")
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    rows = []
    n_total = n_found = 0
    for corpus in CORPORA:
        df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
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
                        rows.append(dict(corpus=corpus, row=i,
                                         reviewer=r.get("reviewer"), model=m,
                                         label=code, position=p / L))
    E = pd.DataFrame(rows)
    E.to_csv(EVENTS, index=False)
    print(f"spans searched {n_total}, located {n_found} "
          f"({100 * n_found / n_total:.2f}%), events after dedupe {len(E)}")
    print(f"wrote {EVENTS}")


if __name__ == "__main__":
    main()
