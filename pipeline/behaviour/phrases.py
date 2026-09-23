#!/usr/bin/env python3
"""Repeated language: stock phrase families, shared phrasebooks, watermark words, and verbatim reuse.

Tables (out/tables/behaviour/):
  phrase_families.csv        curated stock phrases: count per model in the 819 annotated replies, in the
                             highlighted text only, and in the full generated corpus (11,522 replies)
  template_sharing.csv       Jaccard overlap of each model pair's 50 most repeated four-word phrases (in spans)
  template_concentration.csv share of a model's highlighted spans containing one of its own top-20 phrases
  fourgram_reuse.csv         share of a speaker's four-word phrases that recur across its own replies, per corpus
  span_word_counts.csv       word counts over unique highlighted spans, per model (for log-odds watermark words)
  distinctive_vocabulary.csv top-10 words per model by log-odds against the other four
  copy_paste_spans.csv       identical highlighted sentences delivered to different users
"""
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from pipeline.common import benchmark as bm
from pipeline.common.paths import CORPORA, MODELS, RESPONSES_DIR, TAB_B

PHRASES = [
    "makes a lot of", "here are a few things", "i want to make sure", "that's okay", "you can call or text",
    "in a way that feels", "you don't have to be", "probably not", "what do you think", "to be honest with you",
    "i hear you", "that matters", "takes a lot of", "is one of the most", "can make a big difference",
    "you're not alone in this", "a few things to consider", "for example:", "something is wrong with you",
    "that's it", "that makes sense", "that's understandable", "that context changes everything", "that's normal",
    "that's exhausting", "that's important", "you don't have to explain everything", "what you're feeling makes sense",
    "of course it hurt", "please, don't apologize", "that is such a profound realization",
    "that sounds incredibly draining", "that's not unreasonable", "what are your thoughts on that",
    "you're correct", "i feel you", "you are correct",
]
RIGHT_FAMILY = ["you're right", "you are right", "you are absolutely right", "you're absolutely right"]
STOP = set("the a an and or but of to in for on with is are was be it this that you your i my me as at if so not have has do does can will would".split())


def low(t):
    return re.sub(r"[\u2019\u2018]", "'", str(t).lower())


def words(t):
    return re.findall(r"[a-z']+", str(t).lower())


def ngrams(t, n=4):
    w = words(t)
    return [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]


def count(phrases, texts):
    return sum(t.count(p) for p in phrases for t in texts)


def main():
    R = bm.replies()
    S = pd.read_csv(TAB_B / "spans_long.csv")
    S["clean"] = S.text.map(lambda t: re.sub(r"\s+", " ", re.sub(r"[*#>_`]", " ", str(t))).strip())
    U = S.drop_duplicates(["corpus", "row", "model", "clean"])              # unique highlighted spans per reply

    annotated = {m: [low(t) for t in R[(R.model == m) & (R.text.str.len() > 0)].text] for m in MODELS}
    highlighted = {m: [low(t) for t in S[S.model == m].text] for m in MODELS}
    full = {m: [] for m in MODELS}
    for corpus in CORPORA:
        df = pd.read_csv(RESPONSES_DIR / f"{corpus}_results.csv", low_memory=False)
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        for m in MODELS:
            full[m] += [low(t) for t in df[f"{m} Output"].dropna()]
    n_full = sum(len(v) for v in full.values()) // len(MODELS)

    rows = []
    for label, fam in [("you're / you are (absolutely) right", RIGHT_FAMILY)] + [(p, [p]) for p in PHRASES]:
        r = dict(phrase=label)
        for m in MODELS:
            r[f"annotated_{m}"] = count(fam, annotated[m])
        for m in MODELS:
            r[f"highlighted_{m}"] = count(fam, highlighted[m])
        for m in MODELS:
            r[f"full_{m}"] = count(fam, full[m])
        r["annotated_total"] = sum(r[f"annotated_{m}"] for m in MODELS)
        r["highlighted_total"] = sum(r[f"highlighted_{m}"] for m in MODELS)
        r["full_total"] = sum(r[f"full_{m}"] for m in MODELS)
        top = max(MODELS, key=lambda m: r[f"annotated_{m}"])
        r["top_model"] = top
        r["top_share"] = round(r[f"annotated_{top}"] / r["annotated_total"], 3) if r["annotated_total"] else np.nan
        r["scale_factor"] = round(r["full_total"] / r["annotated_total"], 1) if r["annotated_total"] else np.nan
        rows.append(r)
    P = pd.DataFrame(rows).sort_values("annotated_total", ascending=False)
    P.to_csv(TAB_B / "phrase_families.csv", index=False)

    # four-word phrases inside highlighted spans: shared phrasebooks and concentration
    cnt = {m: Counter() for m in MODELS}
    for m in MODELS:
        for sp in U[U.model == m].clean.drop_duplicates():
            cnt[m].update(set(ngrams(sp)))
    top50 = {m: {p for p, _ in cnt[m].most_common(50)} for m in MODELS}
    rows = []
    for a, b in combinations(MODELS, 2):
        inter = top50[a] & top50[b]
        rows.append(dict(model_a=a, model_b=b, jaccard_top50=round(len(inter) / len(top50[a] | top50[b]), 3),
                         n_shared=len(inter),
                         shared_examples=" | ".join(sorted(inter, key=lambda p: -(cnt[a][p] + cnt[b][p]))[:4])))
    pd.DataFrame(rows).sort_values("jaccard_top50", ascending=False).to_csv(TAB_B / "template_sharing.csv", index=False)
    rows = []
    for m in MODELS:
        top20 = [p for p, _ in cnt[m].most_common(20)]
        spans = [set(ngrams(sp)) for sp in U[U.model == m].clean.drop_duplicates()]
        spans = [g for g in spans if g]
        rows.append(dict(model=m, n_spans=len(spans),
                         top20_phrase_coverage=round(np.mean([any(p in g for p in top20) for g in spans]), 3)))
    pd.DataFrame(rows).to_csv(TAB_B / "template_concentration.csv", index=False)

    # four-gram self-reuse across a speaker's own replies, per corpus
    def reuse(texts):
        c = Counter()
        for t in texts:
            c.update(ngrams(t))
        tot = sum(c.values())
        return 100 * sum(v for v in c.values() if v > 1) / tot if tot else np.nan
    rows = []
    for (corpus, m), g in R[R.text.str.len() > 0].groupby(["corpus", "model"]):
        rows.append(dict(corpus=corpus, model=m, pct_4grams_reused=round(reuse(g.text), 1)))
    pd.DataFrame(rows).pivot(index="model", columns="corpus", values="pct_4grams_reused") \
        .reindex(MODELS + ["Human"]).to_csv(TAB_B / "fourgram_reuse.csv")

    # words in unique highlighted spans; log-odds watermark words
    wc = {m: Counter() for m in MODELS}
    for m in MODELS:
        for sp in U[U.model == m].clean:
            wc[m].update(words(sp))
    W = pd.DataFrame(wc).fillna(0).astype(int)
    W.index.name = "word"
    W.to_csv(TAB_B / "span_word_counts.csv")
    N = W.sum()
    rows = []
    for m in MODELS:
        a, b = W[m], W.drop(columns=m).sum(axis=1)
        na, nb = N[m], N.drop(m).sum()
        keep = (a >= 20) & ~W.index.isin(STOP) & (W.index.str.len() > 2)
        lo = np.log((a + 0.5) / (na - a + 0.5)) - np.log((b + 0.5) / (nb - b + 0.5))
        top = lo[keep].sort_values(ascending=False).head(10)
        rows.append(dict(model=m, distinctive_words=", ".join(f"{w}({int(a[w])})" for w in top.index)))
    pd.DataFrame(rows).to_csv(TAB_B / "distinctive_vocabulary.csv", index=False)

    # identical highlighted sentences (>= 6 words) delivered to different users
    rep = U.assign(n_words=U.clean.str.split().str.len())
    rep = rep[rep.n_words >= 6].groupby(["model", "clean"]).row.nunique().rename("n_prompts").reset_index()
    rep = rep[rep.n_prompts > 1].sort_values("n_prompts", ascending=False)
    rep.groupby("model").head(5).rename(columns={"clean": "text"}).to_csv(TAB_B / "copy_paste_spans.csv", index=False)

    print(f"full corpus: {n_full} replies per model")
    print(P[["phrase", "annotated_total", "top_model", "top_share", "full_total", "scale_factor"]].head(12).to_string(index=False))
    print("\nshared top-50 phrases\n", pd.read_csv(TAB_B / "template_sharing.csv").drop(columns="shared_examples").to_string(index=False))
    print("\nfour-gram reuse (%)\n", pd.read_csv(TAB_B / "fourgram_reuse.csv", index_col=0).to_string())


if __name__ == "__main__":
    main()
