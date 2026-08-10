import json
import re
import unicodedata

import numpy as np
import pandas as pd

from paths import (CODES, CORPORA, DATA_DIR, DERIVED, EVENTS, MODNUM, OUT)

import script_metric as sm

MODELS = {n: m for m, n in MODNUM.items()}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("—", "--").replace("–", "-")
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------- extraction
rows = []
n_spans_total = n_found = 0
for corpus in CORPORA:
    df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    for i, r in df.iterrows():
        for n, m in MODELS.items():
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
                    n_spans_total += 1
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
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
print(f"[extract] spans searched {n_spans_total}, found {n_found} "
      f"({100 * n_found / n_spans_total:.2f}%), events after dedupe {len(E)}")

report = {}

# ---------------------------------------- V0: agreement with stage 01's events
# Stage 01 and this file extract the span layer with independent code. They
# must land on the same event set; anything else means one of them is wrong.
print("\n-- V0 cross-check against 01_extract_events.py --")
if EVENTS.exists():
    E01 = pd.read_csv(EVENTS)
    same_total = len(E01) == len(E)
    per_model = (E01.model.value_counts().sort_index()
                 .equals(E.model.value_counts().sort_index()))
    print(f"   stage 01: {len(E01)} events | this file: {len(E)} events | "
          f"totals {'MATCH' if same_total else 'DIFFER'}, "
          f"per-model {'MATCH' if per_model else 'DIFFER'}")
    if not (same_total and per_model):
        print("   WARNING: the two extractions disagree — do not trust either "
              "until the difference is explained.")
    report["V0"] = dict(n_stage01=int(len(E01)), n_here=int(len(E)),
                        totals_match=bool(same_total),
                        per_model_match=bool(per_model))
else:
    print(f"   {EVENTS} not found — run 01_extract_events.py first for V0")

# ------------------------------------------- V1: per-model pooled + V3 control
print("\n-- V1 per-model pooled (200 shuffles, 10 bins) --")
profiles = {}
v1 = []
for m in MODELS.values():
    d = E[E.model == m][["response_id", "label", "position"]].copy()
    d = d.sort_values(["response_id", "position"]).reset_index(drop=True)
    res, prof = sm.compute(d, n_bins=10, n_shuffles=200, seed=0)
    profiles[m] = prof
    v1.append(dict(model=m, SCRIPT=res["SCRIPT"], z=res["z"],
                   C=res["C_excess"], M=res["M_excess"], n=res["n_events"],
                   n_responses=res["n_responses"]))
    print(v1[-1])
report["V1"] = v1

# The headline per-model table. Written here, from this file's own extraction,
# so nothing downstream reads a number this repository did not compute.
pd.DataFrame([dict(system=r["model"], SCRIPT=r["SCRIPT"], z=r["z"],
                   C_excess=r["C"], M_excess=r["M"], n_events=r["n"],
                   n_responses=r["n_responses"], note="counseling codes")
              for r in v1]).to_csv(DERIVED / "validation_results.csv", index=False)
print(f"wrote {DERIVED / 'validation_results.csv'}")

# ----------------------------------------------- V2: the same score per corpus
print("\n-- V2 per-model x per-corpus --")
v2 = []
for m in MODELS.values():
    for c in CORPORA:
        d = (E[(E.model == m) & (E.corpus == c)][["response_id", "label", "position"]]
             .sort_values(["response_id", "position"]).reset_index(drop=True))
        res, _ = sm.compute(d, n_bins=10, n_shuffles=200, seed=0)
        v2.append(dict(model=m, corpus=c, SCRIPT=res["SCRIPT"], z=res["z"],
                       C_excess=res["C_excess"], M_excess=res["M_excess"],
                       n_events=res["n_events"], n_responses=res["n_responses"]))
        print(v2[-1])
report["V2"] = v2
# carry the pooled column in the same table, as corpus "ALL"
v2_all = v2 + [dict(model=r["model"], corpus="ALL", SCRIPT=r["SCRIPT"],
                    z=r["z"], C_excess=r["C"], M_excess=r["M"],
                    n_events=r["n"], n_responses=r["n_responses"]) for r in v1]
pd.DataFrame(v2_all).to_csv(DERIVED / "script_by_corpus.csv", index=False)
print(f"wrote {DERIVED / 'script_by_corpus.csv'}")

print("\n-- V3 random-label control (one model's events, labels shuffled globally) --")
d = E[E.model == "Claude"][["response_id", "label", "position"]].copy()
rng = np.random.default_rng(1)
d["label"] = rng.permutation(d.label.to_numpy())
res, _ = sm.compute(d.sort_values(["response_id", "position"]).reset_index(drop=True),
                    n_bins=10, n_shuffles=200, seed=0)
print(res)
report["V3"] = res

# --------------------------------------------------- V4 synthetic ground truth
print("\n-- V4 synthetic ground truth --")
rng = np.random.default_rng(7)


def synth(kind, n_resp=800, n_ev=6):
    """Corpora with known generating processes.

    template : three labels always in fixed thirds of the reply (pure slot
               structure -> the signal must land in C)
    random   : same label mix, positions and labels independent (must be ~0)
    chain    : labels repeat the previous label w.p. 0.85, positions uniform
               (pure sequential structure -> the signal must land in M)
    """
    rows = []
    for r in range(n_resp):
        if kind == "template":
            evs = [("E", rng.uniform(0, .33)), ("E", rng.uniform(0, .33)),
                   ("A", rng.uniform(.34, .66)), ("A", rng.uniform(.34, .66)),
                   ("Q", rng.uniform(.67, 1)), ("Q", rng.uniform(.67, 1))]
        elif kind == "random":
            evs = [(rng.choice(list("EAQ")), rng.uniform(0, 1)) for _ in range(n_ev)]
        elif kind == "chain":
            evs = sorted([(None, rng.uniform(0, 1)) for _ in range(n_ev)],
                         key=lambda t: t[1])
            labs, prev = [], None
            for _ in evs:
                lab = prev if (prev and rng.random() < .85) else rng.choice(list("EAQ"))
                labs.append(lab)
                prev = lab
            evs = [(labs[i], evs[i][1]) for i in range(len(evs))]
        for lab, x in evs:
            rows.append(dict(response_id=r, label=lab, position=x))
    return pd.DataFrame(rows).sort_values(["response_id", "position"]).reset_index(drop=True)


v4 = {}
for kind in ["template", "random", "chain"]:
    res, _ = sm.compute(synth(kind), n_bins=10, n_shuffles=100, seed=0)
    v4[kind] = {k: res[k] for k in ("SCRIPT", "z", "C_excess", "M_excess")}
    print(kind, v4[kind])
report["V4"] = v4

# ------------------------------------------------------- V5 JS distance matrix
print("\n-- V5 profile JS distances --")
ms = list(MODELS.values())
D = pd.DataFrame(0.0, index=ms, columns=ms)
for i, a in enumerate(ms):
    for b in ms[i + 1:]:
        d_ = sm.profile_distance(profiles[a], profiles[b])
        D.loc[a, b] = D.loc[b, a] = round(d_, 4)
print(D)
report["V5"] = D.to_dict()

# --------------------------------------- V6 held-out corpus identification
print("\n-- V6 identification: enroll on 3 corpora, probe = carebench --")
enroll_prof, probe_dfs = {}, {}
for m in MODELS.values():
    tr = E[(E.model == m) & (E.corpus != "carebench")][["response_id", "label", "position"]]
    te = E[(E.model == m) & (E.corpus == "carebench")][["response_id", "label", "position"]]
    _, prof = sm.compute(tr.sort_values(["response_id", "position"]).reset_index(drop=True),
                         n_bins=10, n_shuffles=5, seed=0)
    enroll_prof[m] = prof
    probe_dfs[m] = te.sort_values(["response_id", "position"]).reset_index(drop=True)
v6 = []
for true_m, probe in probe_dfs.items():
    ll = {m: sm.profile_loglik(probe, enroll_prof[m]) for m in MODELS.values()}
    ranked = sorted(ll.items(), key=lambda kv: -kv[1])
    _, probe_prof = sm.compute(probe, n_bins=10, n_shuffles=5, seed=0)
    js = {m: sm.profile_distance(probe_prof, enroll_prof[m]) for m in MODELS.values()}
    js_pick = min(js, key=js.get)
    v6.append(dict(true=true_m, loglik_pick=ranked[0][0],
                   loglik_margin=round(ranked[0][1] - ranked[1][1], 4),
                   js_pick=js_pick,
                   js_margin=round(sorted(js.values())[1] - min(js.values()), 4)))
    print(v6[-1])
report["V6"] = v6

# ---------------------------------------------- V7 bins + annotator subsets
print("\n-- V7a bin sensitivity (pooled per model) --")
v7a = {}
for nb in (5, 20):
    for m in MODELS.values():
        d = E[E.model == m][["response_id", "label", "position"]]
        res, _ = sm.compute(d.sort_values(["response_id", "position"]).reset_index(drop=True),
                            n_bins=nb, n_shuffles=60, seed=0)
        v7a[f"{m}_bins{nb}"] = (res["SCRIPT"], res["z"])
        print(f"bins={nb} {m}: SCRIPT={res['SCRIPT']} z={res['z']}")
report["V7a"] = v7a

print("\n-- V7b per-annotator subsets (pooled models per annotator) --")
v7b = {}
for rev, g in E.groupby("reviewer"):
    if len(g) < 300:
        print(f"{rev}: only {len(g)} events, skipped")
        continue
    scripts = {}
    for m in MODELS.values():
        d = g[g.model == m][["response_id", "label", "position"]]
        if len(d) < 200:
            continue
        res, _ = sm.compute(d.sort_values(["response_id", "position"]).reset_index(drop=True),
                            n_bins=10, n_shuffles=60, seed=0)
        scripts[m] = (res["SCRIPT"], res["z"])
    v7b[str(rev)] = scripts
    print(rev, scripts)
report["V7b"] = v7b

with open(OUT / "verify_report.json", "w") as f:
    json.dump(report, f, indent=1, default=str)
print(f"\nsaved {OUT / 'verify_report.json'}")
