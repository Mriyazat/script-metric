#!/usr/bin/env python3
"""The worked example of the project website, computed end to end.

One clinician-annotated response (Testbed 1, hope corpus, row 298, Claude) is taken
from the raw annotation file to the score: highlights -> events -> slots and
transitions -> the system's pooled count tables -> the two terms in bits -> the
within-response shuffle null -> SCRIPT, z, and the fraction of the matched ceiling
-> the same response scored under the five enrolled profiles.

Writes docs/example.json and refreshes the inline copy inside docs/index.html
(between the <script id="D"> tags), so the website shows exactly what this
script computes.  Usage: python -m pipeline.figures.web_example
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from pipeline.common.paths import CODES, DATA_DIR, EVENTS, MODELS, MODNUM, REPO

from scriptmetric import metric as sm

CORPUS, ROW, SYSTEM = "hope", 298, "Claude"
B, S, SEED = 10, 200, 0
DOCS = REPO / "docs"

G3 = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
      **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
      **{c: "questions" for c in ["QOP", "QCL"]}}
GROUP = {c: G3.get(c, "other") for c in CODES}
CODE_ORDER = (["VIN", "NIN", "ASIN", "SIN", "VAC", "NAC", "ASAC", "SAC"]
              + ["DIR", "FIX", "RECT"] + ["QOP", "QCL"]
              + ["SEN", "AUR", "TEN", "TSH", "LMT", "MEN", "INC"])
GLOSS = {"VAC": "validation, accurate", "VIN": "validation, inaccurate",
         "NAC": "normalising, accurate", "NIN": "normalising, inaccurate",
         "ASAC": "autonomy support, accurate", "ASIN": "autonomy support, inaccurate",
         "SAC": "support, accurate", "SIN": "support, inaccurate",
         "DIR": "directive", "FIX": "fix-it", "RECT": "recommendation",
         "QOP": "open question", "QCL": "closed question",
         "SEN": "sensitivity", "AUR": "assumes user accuracy", "TEN": "tentative",
         "TSH": "topic shift", "LMT": "language matching",
         "MEN": "minimal encourager", "INC": "incoherent"}


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def located_spans(row, columns, model):
    txt = norm(row.get(f"{model} Output"))
    L = len(txt)
    seen, spans = set(), []
    for code in CODES:
        col = f"Response {MODNUM[model]}_{code}"
        if col not in columns:
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
            spans.append(dict(code=code, group=GROUP[code], p=p, end=p + len(spn),
                              x=round(p / L, 4), X=min(int(p / L * B), B - 1)))
    spans.sort(key=lambda s: (s["p"], CODE_ORDER.index(s["code"])))
    return txt, L, spans


def count_tables(d, labels):
    l2i = {l: i for i, l in enumerate(labels)}
    d = d.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
    lab = d.label.map(l2i).to_numpy()
    xb = np.minimum((d.position.to_numpy() * B).astype(int), B - 1)
    rid = pd.factorize(d.response_id)[0]
    pos = np.zeros((len(labels), B))
    np.add.at(pos, (lab, xb), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = sm._transition_mask(rid, d.position.to_numpy(), "exclude")
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return pos, tr, lab, xb, rid, d.position.to_numpy()


def bits(lab, xb, rid, posv, n_lab):
    """H(L), I(L;X), I(L;L_prev|X) in bits (the quantities C and M are shares of)."""
    H_L = sm._H(np.bincount(lab, minlength=n_lab).astype(float))
    C, M, _ = sm._metrics(lab, xb, rid, n_lab, B, posv, "exclude")
    return H_L, C * H_L, M * H_L


# ------------------------------------------------------------------ the response
df = pd.read_csv(DATA_DIR / f"{CORPUS}_annotated.csv", low_memory=False)
df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
row = df.iloc[ROW]
text, L, events = located_spans(row, df.columns, SYSTEM)

slots = []
for e in events:
    if slots and slots[-1]["p"] == e["p"]:
        slots[-1]["codes"].append(e["code"])
    else:
        slots.append(dict(p=e["p"], x=e["x"], X=e["X"], codes=[e["code"]]))
# slot convention: a transition joins the last code of one slot to the first of the next
trans = [[slots[k]["codes"][-1], slots[k + 1]["codes"][0], slots[k + 1]["X"]]
         for k in range(len(slots) - 1)]

# three within-response re-deals of this reply's own labels over its own positions
rng = np.random.default_rng(SEED)
shuffles = []
for _ in range(3):
    perm = rng.permutation(len(events))
    shuffles.append([dict(code=events[j]["code"], group=events[j]["group"], x=e["x"])
                     for e, j in zip(events, perm)])

# ---------------------------------------------------------------- pooled tables
E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
this_id = f"{CORPUS}|{ROW}|{SYSTEM}"
sysE = E[E.model == SYSTEM]
pos, tr, lab, xb, rid, posv = count_tables(sysE, CODE_ORDER)
H_L, I_LX, I_cond = bits(lab, xb, rid, posv, len(CODE_ORDER))

# null means of the same two quantities, same shuffles as compute(seed=0)
rng = np.random.default_rng(SEED)
null = np.array([sm._metrics(sm._shuffle_within(lab, rid, rng), xb, rid,
                             len(CODE_ORDER), B, posv, "exclude")[:2]
                 for _ in range(S)]) * H_L
nI_LX, nI_cond = null.mean(0)

sys_df = (sysE[["response_id", "label", "position"]]
          .sort_values(["response_id", "position"], kind="stable").reset_index(drop=True))
score, profile = sm.compute(sys_df, B, S, SEED)
ceiling = sm.matched_ceiling(sys_df, B, S, SEED)["SCRIPT"]
sd = score["SCRIPT"] / score["z"]

# -------------------------------------------------------------- identification
probe = sys_df[sys_df.response_id == this_id]
attrib = {}
for m in MODELS:
    d = E[E.model == m][["response_id", "label", "position"]]
    if m == SYSTEM:                                  # enrolled without this response
        d = d[d.response_id != this_id]
    d = d.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
    st, pr = sm.compute(d, B, 50, SEED)
    attrib[m] = dict(loglik=round(float(sm.profile_loglik(probe, pr)), 4),
                     SCRIPT=st["SCRIPT"], n=int(len(d)))
best = max(attrib, key=lambda m: attrib[m]["loglik"])
runner = sorted(attrib.values(), key=lambda a: -a["loglik"])[1]["loglik"]

# ------------------------------------------------------------------------ write
D = dict(
    about="Worked example of the SCRIPT website, computed by pipeline/figures/web_example.py",
    user=norm(row["User Input"]), text=text, L=L,
    labels=CODE_ORDER, fam=[GROUP[c] for c in CODE_ORDER], gloss=GLOSS,
    events=events, slots=slots, trans=trans, shuffles=shuffles,
    pos=pos.astype(int).tolist(), tr=tr.astype(int).tolist(),
    bits=dict(HL=round(H_L, 4), ILX=round(I_LX, 4), ICO=round(I_cond, 4),
              nILX=round(float(nI_LX), 4), nICO=round(float(nI_cond), 4),
              nev=int(pos.sum()), ntr=int(tr.sum())),
    score=dict(R=score["R_raw"], Rnull=score["R_null"], sd=round(sd, 4),
               SCRIPT=score["SCRIPT"], z=score["z"],
               C=score["C_excess"], M=score["M_excess"],
               ceiling=round(float(ceiling), 4),
               frac=round(score["SCRIPT"] / ceiling, 3)),
    attrib=attrib,
    meta=dict(reply=f"{CORPUS}|{ROW}", system=SYSTEM, n_responses=score["n_responses"],
              n_codes=len(CODE_ORDER), B=B, S=S, seed=SEED,
              points_to=best, margin=round(attrib[best]["loglik"] - runner, 3)),
)

DOCS.mkdir(exist_ok=True)
(DOCS / "example.json").write_text(json.dumps(D))
index = DOCS / "index.html"
if index.exists():
    html = index.read_text()
    new, n = re.subn(r'(<script id="D" type="application/json">).*?(</script>)',
                     lambda m: m.group(1) + json.dumps(D) + m.group(2), html, flags=re.S)
    if n:
        index.write_text(new)

print(f"{this_id}: {len(events)} highlights -> {len(slots)} slots -> {len(trans)} transitions")
print(f"H(L) = {H_L:.4f} bits   I(L;X) = {I_LX:.4f}   I(L;L_prev|X) = {I_cond:.4f}")
print(f"C = {I_LX / H_L:.4f}  M = {I_cond / H_L:.4f}  R = {score['R_raw']:.4f}  "
      f"null R = {score['R_null']:.4f} (sd {sd:.4f})")
print(f"SCRIPT = {score['SCRIPT']:.4f}  z = {score['z']}  C excess = {score['C_excess']:.4f}  "
      f"M excess = {score['M_excess']:.4f}  ceiling = {ceiling:.4f}  fraction = {D['score']['frac']:.3f}")
print("log-likelihood under each enrolled profile:",
      {m: attrib[m]["loglik"] for m in MODELS}, "-> points to", best)
print(f"wrote {DOCS / 'example.json'}" + (" and refreshed docs/index.html" if index.exists() else ""))
