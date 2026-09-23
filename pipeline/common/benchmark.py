"""Loaders for the Cognitive Atrophy Benchmark CSVs and the text normalisation
shared by every stage that locates highlighted spans inside a reply."""
import re
import unicodedata

import numpy as np
import pandas as pd

from pipeline.common.paths import (ATTRS, CODES, CORPORA, DATA_DIR, FLAGS, MODELS, MODNUM,
                                   MULTITURN_CORPORA, PROMPT_COL, THERAPIST_COL, USER_ATTRS)


def norm(s) -> str:
    """Lower-case, NFKC, straight quotes, single spaces. Applied identically to a
    reply and to a highlight, so the highlight can be located by substring search."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_md(s: str) -> str:
    return re.sub(r"[*#>_`]", "", s)


def load_annotated(corpus: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    return df


def score_value(cell):
    """Attribute scores are integers; a few cells hold two annotator values as
    ``a|b``, for which the maximum is taken."""
    if pd.isna(cell):
        return np.nan
    vals = pd.to_numeric(pd.Series(str(cell).split("|")), errors="coerce").dropna()
    return float(vals.max()) if len(vals) else np.nan


def items() -> pd.DataFrame:
    """One row per annotated item: corpus, row, reviewer, turn structure, user attributes."""
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            rows.append(dict(corpus=corpus, row=i, reviewer=r.get("reviewer"),
                             turn_type="multi" if corpus in MULTITURN_CORPORA else "single",
                             conversation=r.get("Conversation"), turn=r.get("Turn"),
                             prompt=str(r.get(PROMPT_COL[corpus], "") or ""),
                             **{u: pd.to_numeric(r.get(u), errors="coerce") for u in USER_ATTRS}))
    return pd.DataFrame(rows)


def scores_long() -> pd.DataFrame:
    """One row per (item, model, attribute) with the clinician score."""
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for m in MODELS:
            n = MODNUM[m]
            for a in ATTRS:
                col = f"Response {n}_{a}_score"
                if col not in df.columns:
                    continue
                rows.append(pd.DataFrame(dict(corpus=corpus, row=df.index, model=m, attribute=a,
                                              value=df[col].map(score_value))))
    return pd.concat(rows, ignore_index=True)


def flags_long() -> pd.DataFrame:
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for m in MODELS:
            n = MODNUM[m]
            for f in FLAGS:
                col = f"Response {n}_{f}"
                if col in df.columns:
                    rows.append(pd.DataFrame(dict(corpus=corpus, row=df.index, model=m, flag=f,
                                                  value=pd.to_numeric(df[col], errors="coerce"))))
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def replies() -> pd.DataFrame:
    """One row per (item, speaker) reply text; the therapist's reply is included
    under model='Human'."""
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m in MODELS:
                rows.append(dict(corpus=corpus, row=i, model=m, text=str(r.get(f"{m} Output") or "")))
            rows.append(dict(corpus=corpus, row=i, model="Human", text=str(r.get(THERAPIST_COL[corpus]) or "")))
    out = pd.DataFrame(rows)
    out["text"] = out.text.where(out.text != "nan", "")
    return out


def spans_long() -> pd.DataFrame:
    """One row per highlighted span: which reply, which code, the text, and the
    normalised start position of the highlight inside the reply (NaN when the
    highlight cannot be located verbatim)."""
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m in MODELS:
                n = MODNUM[m]
                reply = str(r.get(f"{m} Output") or "")
                txt = norm(reply)
                txt_md = strip_md(txt)
                L = len(txt)
                for code in CODES:
                    v = r.get(f"Response {n}_{code}")
                    if pd.isna(v) or not str(v).strip():
                        continue
                    for sp in str(v).split(" | "):
                        sp = sp.strip()
                        if not sp or norm(sp) == "#name?":
                            continue
                        spn = norm(sp)
                        p = txt.find(spn)
                        located = p >= 0
                        if not located:
                            located = bool(spn) and strip_md(spn) in txt_md
                        rows.append(dict(corpus=corpus, row=i, reviewer=r.get("reviewer"), model=m, code=code,
                                         text=sp, words=len(sp.split()), located=located,
                                         position=p / L if p >= 0 and L else np.nan, reply_words=len(reply.split())))
    return pd.DataFrame(rows)
