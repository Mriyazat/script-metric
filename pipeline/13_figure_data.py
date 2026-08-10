"""Data prep for the explanatory figures (26-29): one JSON per figure, cached
in out/derived/figdata/ so the figure scripts are pure plotting.

  fig_reply.json    one real annotated Claude reply (hope row 298): text, user
                    message, and every located span with code/group/position
  fig_pooled.json   Claude's 4-group position & transition tables + the pooled
                    20-code score (the worked example's destination number)
  faithful.json     the same at full resolution: 20x10 position table, 20x20
                    transition table, family map, same score
  hero_data.json    one prompt (carebench row 30) answered by all five models:
                    per-model span events, pooled per-model scores, and
                    10-bin position profiles per behaviour group
  mom.json          momentum walkthrough data: what follows VIN vs what
                    position alone predicts, plus the top pulls overall

Needs the downloaded corpora; run 00_get_data.py first.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from paths import CODES, CORPORA, DATA_DIR, EVENTS, MODELS, MODNUM, REF

import script_metric as sm

OUT = REF / "figdata"
OUT.mkdir(exist_ok=True)

G3 = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
      **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
      **{c: "questions" for c in ["QOP", "QCL"]}}
GROUP = {c: G3.get(c, "other") for c in CODES}
GG = ["empathy", "advice", "questions", "other"]
# family-blocked code order for the full-resolution tables
CODE_ORDER = (["VIN", "NIN", "ASIN", "SIN", "VAC", "NAC", "ASAC", "SAC"]
              + ["DIR", "FIX", "RECT"] + ["QOP", "QCL"]
              + ["SEN", "AUR", "TEN", "TSH", "LMT", "MEN", "INC"])


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def located_spans(row, df_columns, model):
    """All spans of one model's reply in one CSV row, located in the text."""
    txt = norm(row.get(f"{model} Output"))
    L = len(txt)
    seen, spans = set(), []
    for code in CODES:
        col = f"Response {MODNUM[model]}_{code}"
        if col not in df_columns:
            continue
        v = row[col]
        if pd.isna(v) or not str(v).strip():
            continue
        for sp in str(v).split(" | "):
            spn = norm(sp)
            if not spn or spn == "#name?":
                continue
            p = txt.find(spn)
            if p < 0 or (code, p) in seen:
                continue
            seen.add((code, p))
            spans.append(dict(code=code, group=GROUP[code], start=p,
                              end=p + len(spn), x=p / L))
    spans.sort(key=lambda s: s["start"])
    return txt, L, spans


def count_tables(df, labels):
    """Row = label: position counts (10 bins) and transition counts."""
    l2i = {l: i for i, l in enumerate(labels)}
    d = df.sort_values(["response_id", "position"]).reset_index(drop=True)
    lab = d.label.map(l2i).to_numpy()
    xb = np.minimum((d.position.to_numpy() * 10).astype(int), 9)
    rid = pd.factorize(d.response_id)[0]
    pos = np.zeros((len(labels), 10))
    np.add.at(pos, (lab, xb), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return pos, tr


E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E["group"] = E.label.map(GROUP)

# ---- fig_reply.json: the worked-example reply (hope row 298, Claude) --------
hope = pd.read_csv(DATA_DIR / "hope_annotated.csv", low_memory=False)
hope.columns = [c.lstrip("\ufeff").strip() for c in hope.columns]
r = hope.iloc[298]
txt, L, spans = located_spans(r, hope.columns, "Claude")
json.dump(dict(user=norm(r["User Input"]), txt=txt, L=L, spans=spans),
          open(OUT / "fig_reply.json", "w"))
print(f"fig_reply: {len(spans)} spans, {L} chars")

# ---- fig_pooled.json + faithful.json: Claude's tables and score -------------
cl = E[E.model == "Claude"]
score, _ = sm.compute(cl[["response_id", "label", "position"]]
                      .sort_values(["response_id", "position"]).reset_index(drop=True),
                      10, 200, 0)
score = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
         for k, v in score.items()}

gpos, gtr = count_tables(cl.assign(label=cl.group), GG)
json.dump(dict(pos=gpos.tolist(), tr=gtr.tolist(), score=score),
          open(OUT / "fig_pooled.json", "w"))

cpos, ctr = count_tables(cl, CODE_ORDER)
json.dump(dict(labels=CODE_ORDER, fam=[GROUP[c] for c in CODE_ORDER],
               pos=cpos.tolist(), tr=ctr.tolist(), score=score),
          open(OUT / "faithful.json", "w"))
print(f"pooled score: SCRIPT={score['SCRIPT']:.3f} C={score['C_excess']:.3f} "
      f"M={score['M_excess']:.3f} z={score['z']:.0f}")

# ---- hero_data.json: one prompt, five replies, five profiles ----------------
care = pd.read_csv(DATA_DIR / "carebench_annotated.csv", low_memory=False)
care.columns = [c.lstrip("\ufeff").strip() for c in care.columns]
row = care.iloc[30]
sample_events = {}
for m in MODELS:
    _, _, sp = located_spans(row, care.columns, m)
    sample_events[m] = [[s["x"], s["group"]] for s in sp if s["group"] != "other"]

scores, sig = {}, {}
for m in MODELS:
    d = E[E.model == m]
    st, _ = sm.compute(d[["response_id", "label", "position"]]
                       .sort_values(["response_id", "position"]).reset_index(drop=True),
                       10, 200, 0)
    scores[m] = dict(SCRIPT=float(st["SCRIPT"]), z=float(st["z"]))
    sig[m] = {}
    for g in ["empathy", "advice", "questions"]:
        x = d[d.group == g].position
        h, _ = np.histogram(x, bins=np.linspace(0, 1, 11))
        sig[m][g] = (h / h.sum()).tolist()
json.dump(dict(user=norm(row["User Input"]),
               sample=dict(events=sample_events), scores=scores, sig=sig),
          open(OUT / "hero_data.json", "w"))
print("hero scores:", {m: round(scores[m]["SCRIPT"], 3) for m in MODELS})

# ---- mom.json: the momentum walkthrough (all-model events, 20 codes) --------
resp = {}
examples = []
for corpus in CORPORA:
    df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    for i, rr in df.iterrows():
        out = rr.get("Claude Output")
        if pd.isna(out):
            continue
        t, l, sp = located_spans(rr, df.columns, "Claude")
        if not sp:
            continue
        resp[f"{corpus}|{i}"] = [(s["x"], min(int(s["x"] * 10), 9), s["code"],
                                  t[s["start"]:s["end"]][:60]) for s in sp]
cidx = {c: k for k, c in enumerate(CODES)}
bincount = np.zeros((10, len(CODES)))
for ev in resp.values():
    for _, b, c, _ in ev:
        bincount[b, cidx[c]] += 1
Pcode_bin = bincount / bincount.sum(1, keepdims=True)

obs = np.zeros(len(CODES))
base = np.zeros(len(CODES))
nvin = 0
for ev in resp.values():
    for k in range(len(ev) - 1):
        if ev[k][2] == "VIN":
            obs[cidx[ev[k + 1][2]]] += 1
            base += Pcode_bin[ev[k + 1][1]]
            nvin += 1
            if len(examples) < 12 and ev[k + 1][2] == "DIR":
                a = ev[k][3].strip(" -\u2013:\u201c\u201d")
                b = ev[k + 1][3].strip(" -\u2013:\u201c\u201d")
                # keep quotes presentable and from fully distinct hand-offs
                # (no text may reappear on either side of another example)
                pool = [t[:20] for ex in examples for t in ex]
                if not any(ch in a + b for ch in "*#>|") \
                        and a[:20] not in pool and b[:20] not in pool:
                    examples.append((a, b))
obs /= obs.sum()
base /= base.sum()

pulls = []
for cur in CODES:
    o = np.zeros(len(CODES))
    bb = np.zeros(len(CODES))
    n = 0
    for ev in resp.values():
        for k in range(len(ev) - 1):
            if ev[k][2] == cur:
                o[cidx[ev[k + 1][2]]] += 1
                bb += Pcode_bin[ev[k + 1][1]]
                n += 1
    if n < 80:
        continue
    o /= o.sum()
    bb /= bb.sum()
    nx = int(np.argmax(o - bb))
    pulls.append((cur, CODES[nx], float((o - bb)[nx]), n))
pulls.sort(key=lambda t: -t[2])
json.dump(dict(codes=CODES, fam=GROUP, obs=obs.tolist(), base=base.tolist(),
               nvin=nvin, examples=examples[:6], pulls=pulls[:8],
               M_excess=score["M_excess"]),
          open(OUT / "mom.json", "w"))
print(f"mom: n_VIN={nvin}, VIN->DIR surplus={obs[cidx['DIR']] - base[cidx['DIR']]:+.3f}")
print("top pulls:", [(a, b, round(s, 3)) for a, b, s, _ in pulls[:5]])
print("saved all to", OUT)
