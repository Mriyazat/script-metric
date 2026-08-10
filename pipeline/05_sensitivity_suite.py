import glob
import json
import sys

import numpy as np
import pandas as pd

from paths import CKPT, CORPORA, DATA_DIR, EVENTS, MODELS, MODNUM, TAB

import script_metric as sm

NB = 10
rng = np.random.default_rng(21)

E = pd.read_csv(EVENTS)
E["unit"] = E.corpus + "|" + E.row.astype(str)
E["response_id"] = E.unit + "|" + E.model
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)


def sc(df, n_shuffles=60, seed=0, nb=NB):
    res, _ = sm.compute(df[["response_id", "label", "position"]]
                        .sort_values(["response_id", "position"]).reset_index(drop=True),
                        n_bins=nb, n_shuffles=n_shuffles, seed=seed)
    return res


STAGE = sys.argv[1]

# ---------------------------------------------------------------- Q1
if STAGE == "Q1":
    # Map each response to its expert ratings. EMP_score is the graded
    # empathy-accuracy judgment (0 = inaccurate, 1 = partial, 2 = accurate);
    # yn_harmful is the binary harmful-content flag. Both are used as
    # categorical strata, never averaged as ordinal severities.
    meta = {}
    for corpus in CORPORA:
        df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        for i, r in df.iterrows():
            for m, n in MODNUM.items():
                emp = r.get(f"Response {n}_EMP_score")
                harm = r.get(f"Response {n}_yn_harmful")
                meta[f"{corpus}|{i}|{m}"] = (emp, harm)
    E["emp"] = E.response_id.map(lambda k: meta.get(k, (np.nan, np.nan))[0])
    E["harm"] = E.response_id.map(lambda k: meta.get(k, (np.nan, np.nan))[1])
    E["emp"] = pd.to_numeric(E.emp, errors="coerce")
    E["harm"] = pd.to_numeric(E.harm, errors="coerce")
    rows = []
    for m in MODELS:
        d = E[E.model == m]
        strata = {"EMP=0 (inaccurate)": d[d.emp == 0],
                  "EMP=2 (accurate)": d[d.emp == 2],
                  "harmful=1": d[d.harm == 1],
                  "harmful=0": d[d.harm == 0]}
        for name, dd in strata.items():
            if dd.response_id.nunique() < 30:
                continue
            res = sc(dd)
            rows.append(dict(model=m, stratum=name, SCRIPT=res["SCRIPT"], z=res["z"],
                             n_events=res["n_events"], n_resp=res["n_responses"]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "script_vs_quality.csv", index=False)

# ---------------------------------------------------------------- Q2
elif STAGE == "Q2":
    # Alternative label alphabets. The groupings are display/robustness
    # partitions of the 20 codes, not risk directions: any partition is a
    # valid alphabet for the metric.
    MERGE13 = {c: c for c in ["SEN", "AUR", "TEN", "DIR", "FIX", "RECT", "TSH",
                              "QOP", "QCL", "LMT", "MEN", "INC"]}
    MERGE13.update({c: "EMP_ACC" for c in ["VAC", "NAC", "ASAC", "SAC"]})
    MERGE13.update({c: "EMP_INA" for c in ["VIN", "NIN", "ASIN", "SIN"]})
    G5 = {**{c: "advice" for c in ["DIR", "FIX", "RECT"]},
          **{c: "quest" for c in ["QOP", "QCL"]},
          **{c: "emp_acc" for c in ["VAC", "NAC", "ASAC", "SAC"]},
          **{c: "emp_ina" for c in ["VIN", "NIN", "ASIN", "SIN"]},
          **{c: "other" for c in ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}}
    SCHEMES = {"20 codes": None, "13 codes (empathy merged)": MERGE13, "5 groups": G5}
    rows = []
    for sname, mp in SCHEMES.items():
        Es = E.copy()
        if mp is not None:
            Es["label"] = Es.label.map(mp)
        # per-model score at this granularity
        for m in MODELS:
            res = sc(Es[Es.model == m])
            rows.append(dict(scheme=sname, model=m, SCRIPT=res["SCRIPT"], z=res["z"],
                             C=res["C_excess"], M=res["M_excess"], n_labels=res["n_labels"]))
            print(rows[-1], flush=True)
        # held-out-corpus identification at this granularity
        enroll, ok_ll, ok_js = {}, 0, 0
        for m in MODELS:
            tr = Es[(Es.model == m) & (Es.corpus != "carebench")]
            _, prof = sm.compute(tr[["response_id", "label", "position"]]
                                 .sort_values(["response_id", "position"]).reset_index(drop=True),
                                 n_bins=NB, n_shuffles=2, seed=0)
            enroll[m] = prof
        for m in MODELS:
            te = Es[(Es.model == m) & (Es.corpus == "carebench")][["response_id", "label", "position"]]
            te = te.sort_values(["response_id", "position"]).reset_index(drop=True)
            ll = {mm: sm.profile_loglik(te, enroll[mm]) for mm in MODELS}
            if max(ll, key=ll.get) == m:
                ok_ll += 1
            _, pp = sm.compute(te, n_bins=NB, n_shuffles=2, seed=0)
            js = {mm: sm.profile_distance(pp, enroll[mm]) for mm in MODELS}
            if min(js, key=js.get) == m:
                ok_js += 1
        rows.append(dict(scheme=sname, model="ID(carebench held out)",
                         SCRIPT=f"{ok_ll}/5 loglik", z=f"{ok_js}/5 JS",
                         C="", M="", n_labels=""))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "granularity.csv", index=False)

# ---------------------------------------------------------------- Q3
elif STAGE == "Q3":
    G5 = {**{c: 0 for c in ["DIR", "FIX", "RECT"]},
          **{c: 1 for c in ["QOP", "QCL"]},
          **{c: 2 for c in ["VAC", "NAC", "ASAC", "SAC"]},
          **{c: 3 for c in ["VIN", "NIN", "ASIN", "SIN"]},
          **{c: 4 for c in ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}}
    NG = 5

    def skip_mi(lab, xbin, rid, gap):
        """I(L; L_prev<gap> | X) / H(L) at group level."""
        counts = np.bincount(lab, minlength=NG).astype(float)
        p = counts[counts > 0] / counts.sum()
        H_L = -(p * np.log2(p)).sum()
        ok = np.ones(len(lab) - gap, bool)
        for g in range(gap):
            ok &= rid[gap:] == rid[g:len(rid) - gap + g]
        cur, prv, xb = lab[gap:][ok], lab[:-gap][ok], xbin[gap:][ok]
        n = len(cur)
        tot = 0.0
        if n == 0:
            return 0.0
        for b in range(NB):
            msk = xb == b
            if msk.sum() < 2:
                continue
            J = np.zeros((NG, NG))
            np.add.at(J, (cur[msk], prv[msk]), 1.0)
            tot += (msk.sum() / n) * (sm._H(J.sum(1)) + sm._H(J.sum(0)) - sm._H(J.ravel()))
        return tot / H_L

    rows = []
    for m in MODELS:
        d = E[E.model == m].sort_values(["response_id", "position"])
        lab = d.label.map(G5).to_numpy()
        xb = np.minimum((d.position.to_numpy() * NB).astype(int), NB - 1)
        rid = pd.factorize(d.response_id)[0]
        r = dict(model=m)
        for gap, nm in [(1, "order1"), (2, "skip1"), (3, "skip2")]:
            v = skip_mi(lab, xb, rid, gap)
            null = np.array([skip_mi(sm._shuffle_within(lab, rid, rng), xb, rid, gap)
                             for _ in range(60)])
            r[nm] = round(v - null.mean(), 4)
            r[nm + "_z"] = round((v - null.mean()) / null.std(), 1)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(TAB / "skipgram.csv", index=False)

# ---------------------------------------------------------------- Q4
elif STAGE == "Q4":
    rows = []
    for m in ["Claude", "Llama"]:
        d = E[E.model == m]
        resp = d.response_id.unique()
        for N in [25, 50, 100, 200, 400, 800]:
            if N > len(resp):
                continue
            vals = []
            for t in range(10):
                sel = np.random.default_rng(100 + t).choice(resp, N, replace=False)
                res = sc(d[d.response_id.isin(sel)], n_shuffles=40)
                vals.append((res["SCRIPT"], res["z"], res["n_events"]))
            V = np.array(vals)
            rows.append(dict(model=m, n_resp=N, SCRIPT_mean=round(V[:, 0].mean(), 4),
                             SCRIPT_sd=round(V[:, 0].std(), 4),
                             z_mean=round(V[:, 1].mean(), 1),
                             mean_events=int(V[:, 2].mean())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "sample_size_curve.csv", index=False)
    rows = []
    for nb in [3, 5, 10, 20, 40]:
        for m in MODELS:
            res = sc(E[E.model == m], n_shuffles=40, nb=nb)
            rows.append(dict(bins=nb, model=m, SCRIPT=res["SCRIPT"], z=res["z"]))
        print("bins", nb, "done", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "bin_sweep.csv", index=False)

# ---------------------------------------------------------------- Q5
elif STAGE.startswith("Q5A"):
    rev = STAGE.split(":")[1]
    sub = E[E.reviewer == rev]
    units = sorted(sub.unit.unique())
    pre, lab_maps = {}, {}
    for m in MODELS:
        d = sub[sub.model == m]
        labels = sorted(d.label.unique())
        l2i = {l: i for i, l in enumerate(labels)}
        lab_maps[m] = labels
        pre[m] = {u: (dd.sort_values("position").label.map(l2i).to_numpy(),
                      np.minimum((dd.sort_values("position").position.to_numpy() * NB).astype(int),
                                 NB - 1))
                  for u, dd in d.groupby("unit")}
    B = 100
    out = np.zeros((B, 5))
    for b in range(B):
        rr = np.random.default_rng(5000 + b)
        samp = rr.choice(units, len(units), replace=True)
        for mi, m in enumerate(MODELS):
            labs, xbs, rids = [], [], []
            rid = 0
            for u in samp:
                arr = pre[m].get(u)
                if arr is None or len(arr[0]) == 0:
                    continue
                labs.append(arr[0])
                xbs.append(arr[1])
                rids.append(np.full(len(arr[0]), rid))
                rid += 1
            if not labs:
                out[b, mi] = np.nan
                continue
            lab = np.concatenate(labs)
            xb = np.concatenate(xbs)
            ri = np.concatenate(rids)
            C, M, R = sm._metrics(lab, xb, ri, len(lab_maps[m]), NB)
            null = np.array([sm._metrics(sm._shuffle_within(lab, ri, rr), xb, ri,
                                         len(lab_maps[m]), NB)[2] for _ in range(15)])
            out[b, mi] = R - null.mean()
    jury_ckpt = CKPT / "jury_ckpt"
    jury_ckpt.mkdir(exist_ok=True)
    np.save(jury_ckpt / f"{rev}.npy", out)
    print(rev, "done")

elif STAGE == "Q5COLLECT":
    rows, gaps, ses = [], [], []
    for f in sorted(glob.glob(str(CKPT / "jury_ckpt" / "*.npy"))):
        rev = f.split("/")[-1][:-4]
        A = np.load(f)
        # Tier gap: mean({Llama, Claude}) - mean({Qwen, GPT, Gemini}),
        # following the MODELS order (Qwen, Llama, GPT, Claude, Gemini).
        gap = (A[:, 1] + A[:, 3]) / 2 - (A[:, 0] + A[:, 2] + A[:, 4]) / 3
        gap = gap[~np.isnan(gap)]
        if len(gap) < 50:
            continue
        g, se = gap.mean(), gap.std()
        rows.append(dict(annotator=rev, gap=round(g, 4), se=round(se, 4)))
        gaps.append(g)
        ses.append(se)
    g, se = np.array(gaps), np.array(ses)
    w = 1 / se ** 2
    fixed = (w * g).sum() / w.sum()
    Qh = (w * (g - fixed) ** 2).sum()
    dfree = len(g) - 1
    Cc = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0, (Qh - dfree) / Cc)
    wr = 1 / (se ** 2 + tau2)
    re = (wr * g).sum() / wr.sum()
    re_se = np.sqrt(1 / wr.sum())
    rows.append(dict(annotator="RE pooled (DL)", gap=round(re, 4), se=round(re_se, 4)))
    T = pd.DataFrame(rows)
    T["z"] = (T.gap / T.se).round(2)
    T.to_csv(TAB / "tier_gap_meta.csv", index=False)
    print(T)
    print(f"tau2={tau2:.6f}  tau={np.sqrt(tau2):.4f}  I2={max(0, (Qh - dfree) / Qh) * 100:.0f}%")
    with open(TAB / "tier_gap_meta_stats.json", "w") as fh:
        json.dump(dict(tau2=tau2, tau=float(np.sqrt(tau2)), Q=float(Qh), df=dfree,
                       I2=float(max(0, (Qh - dfree) / Qh)), RE_gap=float(re),
                       RE_se=float(re_se), RE_z=float(re / re_se)), fh, indent=1)

print("stage done")
