#!/usr/bin/env python3
"""Span-by-span agreement between the blind LLM annotation layer and the clinicians.

For every clinician highlight on a model reply, the highlighted characters are
compared with the LLM layer's highlights on the same reply (from the per-reply
cache written by external/llm_annotator.py):

    text overlap   share of clinician-highlighted characters that the LLM also
                   highlighted under *any* code;
    code overlap   share highlighted by the LLM under the *same* code;
    per code       share of clinician spans of code c that are matched (>= half
                   of their characters) by an LLM span of code c.

    python -m pipeline.external.llm_agreement [--annotator deepseek-v4-pro]

Writes out/tables/llm_agreement.csv (per code) and out/tables/llm_agreement_summary.csv."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from pipeline.common.benchmark import load_annotated, norm
from pipeline.common.paths import CODES, CORPORA, DERIVED, MODELS, MODNUM, TAB

CACHE = DERIVED / "llm_annotator"


def clinician_spans(df, r, m):
    """(code, start, end) of every located clinician highlight on reply m of row r."""
    txt = norm(r.get(f"{m} Output"))
    out = []
    n = MODNUM[m]
    for code in CODES:
        col = f"Response {n}_{code}"
        if col not in df.columns:
            continue
        v = r[col]
        if pd.isna(v) or not str(v).strip():
            continue
        seen = set()
        for sp in str(v).split(" | "):
            spn = norm(sp)
            if not spn or spn == "#name?":
                continue
            p = txt.find(spn)
            if p < 0 or (code, p) in seen:
                continue
            seen.add((code, p))
            out.append((code, p, p + len(spn)))
    return txt, out


def llm_spans(cache_dir, corpus, row, m, txt):
    f = cache_dir / f"{corpus}_{row}_{m}.json"
    if not f.exists():
        return None
    out = []
    for sp in json.loads(f.read_text())["spans"]:
        q = norm(sp["quote"])
        if not q:
            continue
        p = txt.find(q)
        if p >= 0:
            out.append((sp["code"], p, p + len(q)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="deepseek-v4-pro")
    args = ap.parse_args()
    cache_dir = CACHE / re.sub(r"[^\w.-]", "_", args.annotator)

    covered_any = covered_same = total_chars = 0
    per_code = {c: [0, 0] for c in CODES}          # matched, total spans
    n_replies = n_clin = n_llm = 0
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m in MODELS:
                out = r.get(f"{m} Output")
                if pd.isna(out):
                    continue
                txt, cs = clinician_spans(df, r, m)
                if not cs:
                    continue
                ls = llm_spans(cache_dir, corpus, i, m, txt)
                if ls is None:
                    continue
                n_replies += 1
                n_clin += len(cs)
                n_llm += len(ls)
                L = len(txt)
                any_mask = np.zeros(L, bool)
                code_mask = {c: np.zeros(L, bool) for c in CODES}
                for code, a, b in ls:
                    any_mask[a:b] = True
                    if code in code_mask:
                        code_mask[code][a:b] = True
                for code, a, b in cs:
                    total_chars += b - a
                    covered_any += int(any_mask[a:b].sum())
                    same = int(code_mask[code][a:b].sum())
                    covered_same += same
                    per_code[code][1] += 1
                    per_code[code][0] += same >= 0.5 * (b - a)
    summary = dict(annotator=args.annotator, replies=n_replies, clinician_spans=n_clin,
                   llm_spans=n_llm, spans_per_reply_clinician=round(n_clin / n_replies, 2),
                   spans_per_reply_llm=round(n_llm / n_replies, 2),
                   text_overlap=round(covered_any / total_chars, 4),
                   code_overlap=round(covered_same / total_chars, 4))
    print(summary)
    rows = [dict(code=c, clinician_spans=t, matched_same_code=k,
                 match_rate=round(k / t, 4) if t else np.nan)
            for c, (k, t) in per_code.items()]
    P = pd.DataFrame(rows).sort_values("match_rate", ascending=False)
    print(P.to_string(index=False))
    P.to_csv(TAB / "llm_agreement.csv", index=False)
    pd.DataFrame([summary]).to_csv(TAB / "llm_agreement_summary.csv", index=False)
    print(f"wrote {TAB / 'llm_agreement.csv'}")


if __name__ == "__main__":
    main()
