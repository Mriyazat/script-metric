import json
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from paths import CODES, CORPORA, DATA_DIR, EVENTS, MODELS, MODNUM, RAGTRUTH_PATH, TAB

import script_metric as sm

NB = 10
rng = np.random.default_rng(31)

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)


def arrays(m):
    d = E[E.model == m]
    labels = sorted(d.label.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab = d.label.map(l2i).to_numpy()
    xb = np.minimum((d.position.to_numpy() * NB).astype(int), NB - 1)
    rid = pd.factorize(d.response_id)[0]
    return lab, xb, rid, len(labels)


def segments(rid):
    order = np.argsort(rid, kind="stable")
    bounds = np.flatnonzero(np.diff(rid[order])) + 1
    return np.split(order, bounds)


STAGE = sys.argv[1]

if STAGE == "NULLS":
    rows = []
    for m in MODELS:
        lab, xb, rid, nl = arrays(m)
        segs = segments(rid)
        C, M, R = sm._metrics(lab, xb, rid, nl, NB)

        def null_R(shuffler, S=100):
            vals = []
            for _ in range(S):
                vals.append(sm._metrics(shuffler(lab), xb, rid, nl, NB))
            return np.array(vals)

        # (a) full within-response permutation (reference null)
        full = null_R(lambda l: sm._shuffle_within(l, rid, rng))

        # (b) circular shift within response (preserves transitions)
        def circ(l):
            out = l.copy()
            for seg in segs:
                n = len(seg)
                if n > 1:
                    out[seg] = out[np.roll(seg, rng.integers(1, n))]
            return out

        cir = null_R(circ)

        # (c) permutation within (response, bin) cells
        #     (preserves the label-bin joint exactly)
        cells = {}
        for i, (r, b) in enumerate(zip(rid, xb)):
            cells.setdefault((r, b), []).append(i)
        cells = [np.array(v) for v in cells.values() if len(v) > 1]

        def binperm(l):
            out = l.copy()
            for c in cells:
                out[c] = out[rng.permutation(c)]
            return out

        binp = null_R(binperm)
        rows.append(dict(model=m,
                         SCRIPT_full=round(R - full[:, 2].mean(), 4),
                         C_excess=round(C - full[:, 0].mean(), 4),
                         excess_circshift=round(R - cir[:, 2].mean(), 4),
                         M_excess=round(M - full[:, 1].mean(), 4),
                         excess_withinbin=round(R - binp[:, 2].mean(), 4),
                         z_circ=round((R - cir[:, 2].mean()) / cir[:, 2].std(), 1),
                         z_bin=round((R - binp[:, 2].mean()) / binp[:, 2].std(), 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "alternative_nulls.csv", index=False)

elif STAGE == "LEN":
    # Re-extract events including span duration (share of the reply length).
    def norm(s):
        s = unicodedata.normalize("NFKC", str(s)).replace("—", "--").replace("–", "-")
        s = (s.replace("’", "'").replace("‘", "'")
              .replace("“", '"').replace("”", '"'))
        return re.sub(r"\s+", " ", s).strip().lower()

    rows_ev = []
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
                        p = txt.find(spn)
                        if p < 0:
                            continue
                        key = (code, p)
                        if key in seen:
                            continue
                        seen.add(key)
                        rows_ev.append((f"{corpus}|{i}|{m}", m, code, p / L,
                                        min(len(spn) / L, 1.0)))
    EV = pd.DataFrame(rows_ev, columns=["response_id", "model", "label", "position", "dur"])
    EV = EV.sort_values(["response_id", "position"]).reset_index(drop=True)
    out = []
    for m in MODELS:
        d = EV[EV.model == m]
        labels = sorted(d.label.unique())
        l2i = {l: i for i, l in enumerate(labels)}
        lab = d.label.map(l2i).to_numpy()
        xb = np.minimum((d.position.to_numpy() * NB).astype(int), NB - 1)
        db = np.minimum((d.dur.to_numpy() * NB).astype(int), NB - 1)   # duration decile
        rid = pd.factorize(d.response_id)[0]
        nl = len(labels)

        def I_LX(l, X, nx):
            J = np.zeros((nl, nx))
            np.add.at(J, (l, X), 1.0)
            return sm._H(J.sum(1)) + sm._H(J.sum(0)) - sm._H(J.ravel())

        counts = np.bincount(lab, minlength=nl).astype(float)
        p = counts[counts > 0] / counts.sum()
        H_L = -(p * np.log2(p)).sum()
        # joint (start-bin, duration-bin) as one variable for I(L; X, D)
        XD = xb * NB + db

        def gain(l):
            return (I_LX(l, XD, NB * NB) - I_LX(l, xb, NB)) / H_L

        g = gain(lab)
        null = np.array([gain(sm._shuffle_within(lab, rid, rng)) for _ in range(80)])
        # midpoint variant of C
        xm = np.minimum(((d.position + d.dur / 2).clip(0, 0.9999).to_numpy() * NB).astype(int),
                        NB - 1)
        Cs = I_LX(lab, xb, NB) / H_L
        Cm = I_LX(lab, xm, NB) / H_L
        nullS = np.array([I_LX(sm._shuffle_within(lab, rid, rng), xb, NB) / H_L
                          for _ in range(80)])
        nullM = np.array([I_LX(sm._shuffle_within(lab, rid, rng), xm, NB) / H_L
                          for _ in range(80)])
        out.append(dict(model=m,
                        C_start=round(Cs - nullS.mean(), 4),
                        C_mid=round(Cm - nullM.mean(), 4),
                        dur_gain=round(g - null.mean(), 4),
                        dur_gain_z=round((g - null.mean()) / null.std(), 1)))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(TAB / "length_ablation.csv", index=False)

elif STAGE == "SHUF":
    rows = []
    for m, corp in [("Claude", None), ("Claude", "counselchat")]:
        d = E[E.model == m] if corp is None else E[(E.model == m) & (E.corpus == corp)]
        dd = d[["response_id", "label", "position"]].sort_values(
            ["response_id", "position"]).reset_index(drop=True)
        for S in [25, 50, 100, 200, 500]:
            reps = 8 if S <= 200 else 4
            vals = []
            for t in range(reps):
                res, _ = sm.compute(dd, n_bins=NB, n_shuffles=S, seed=1000 + t)
                vals.append((res["SCRIPT"], res["z"]))
            V = np.array(vals)
            rows.append(dict(setting=f"{m}/{corp or 'pooled'}", S=S,
                             SCRIPT_mean=round(V[:, 0].mean(), 4),
                             SCRIPT_sd=round(V[:, 0].std(), 5),
                             z_mean=round(V[:, 1].mean(), 1),
                             z_sd=round(V[:, 1].std(), 2)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "shuffle_count_stability.csv", index=False)

elif STAGE == "CARD":
    rows = []
    for m in ["Claude"]:
        d = E[E.model == m][["response_id", "label", "position"]]
        res, _ = sm.compute(d.sort_values(["response_id", "position"]).reset_index(drop=True),
                            n_bins=NB, n_shuffles=100, seed=0)
        counts = d.label.value_counts(normalize=True)
        H_L = float(-(counts * np.log2(counts)).sum())
        rows.append(dict(scheme="counseling codes", system=m, n_labels=res["n_labels"],
                         H_L=round(H_L, 2), n_events=res["n_events"],
                         R_null=res["R_null"], SCRIPT=res["SCRIPT"], z=res["z"]))
    try:
        recs = [json.loads(l) for l in open(RAGTRUTH_PATH)]
        for gen in ["gpt-4-0613", "mistral-7B-instruct"]:
            ev = []
            for r in recs:
                if r["model"] != gen or not r["labels"]:
                    continue
                L = len(r["response"])
                if L == 0:
                    continue
                for lb in r["labels"]:
                    ev.append(dict(response_id=r["id"], label=lb["label_type"],
                                   position=min(lb["start"] / L, 1.0)))
            df = pd.DataFrame(ev).sort_values(["response_id", "position"]).reset_index(drop=True)
            res, _ = sm.compute(df, n_bins=NB, n_shuffles=100, seed=0)
            counts = df.label.value_counts(normalize=True)
            H_L = float(-(counts * np.log2(counts)).sum())
            rows.append(dict(scheme="RAGTruth hallucination", system=gen,
                             n_labels=res["n_labels"], H_L=round(H_L, 2),
                             n_events=res["n_events"], R_null=res["R_null"],
                             SCRIPT=res["SCRIPT"], z=res["z"]))
    except FileNotFoundError:
        print(f"RAGTruth file not found at {RAGTRUTH_PATH} — RAGTruth rows skipped.")
    pd.DataFrame(rows).to_csv(TAB / "calibration_cards.csv", index=False)
    print(pd.DataFrame(rows))

print("stage done")
