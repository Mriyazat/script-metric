#!/usr/bin/env python3
"""Sensitivity of the estimator and of the extensions (Appendix "Sensitivity").

Stages (run from the repository root; each writes to out/tables/reviewer_qs/):

    BINS      bin count and binning scheme (fixed-width vs quantile), with the
              matched ceiling at every B                       -> q1_bin_sweep.csv
    PERSON    coverage, entropy and capacity-normalised shares of the person
              term (LLM layer, multi-turn corpora)              -> q3_person_budget.csv
    EXTENT    span start / midpoint / end, and the span-length decile as a
              conditioning variable                             -> q4_span_extent.csv
    ORDER2    order-2 momentum gain at 5 groups and 20 codes    -> q5_order2.csv
    DENSITY   events per reply: synthetic template and chain at four densities,
              reply-length terciles with the matched ceiling    -> q6_synthetic_density.csv,
                                                                   q6_reply_length.csv,
                                                                   q6_fraction_by_length.csv
    SEQPOWER  SCRIPT-Seq stopping times on real and strength-controlled
              synthetic streams                                 -> q7_real_power.csv,
                                                                   q7_synthetic_power.csv
    TAGGER    position dependence of the Testbed-2 tagger: lexical proxy by
              position fifth, and a position-blind rule labeller -> q2_tagger_position_*.csv,
                                                                   q2_lexical_labeller.csv
    PROFILES  what moves a profile: JS distances across the 22 Testbed-2
              systems and a 40-turn probe                       -> q8_js_matrix.csv, q8_systems.csv
    ALL       everything above in order

All SCRIPT values use scriptmetric.metric.compute with its defaults (10 bins, slot
convention). Requires out/derived/span_events.csv; PERSON, ORDER2 also need
out/derived/llm_span_events.csv; TAGGER, PROFILES need the Zhan et al. release
under raw/mint-empathy and out/tables/mint_systems.csv."""
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd

from pipeline.common.benchmark import load_annotated, norm
from pipeline.common.paths import CODES, CORPORA, DERIVED, EVENTS, MINT_DIR, MODELS, MODNUM, TAB

from scriptmetric import metric as sm
from scriptmetric.metric import _H, _shuffle_within

warnings.filterwarnings("ignore")
OUT = TAB / "reviewer_qs"
OUT.mkdir(parents=True, exist_ok=True)

UCOLS = ["user_sensitivity", "user_evocative", "user_typicality", "user_underlying", "user_request_info"]


def load_events(path):
    E = pd.read_csv(path)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    return E.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)


def sub(df, m):
    return df[df.model == m][["response_id", "label", "position"]].reset_index(drop=True)


def _mi(a, b):
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    return _H(ct.sum(1)) + _H(ct.sum(0)) - _H(ct.ravel())


def slot_predecessor(rid, pos):
    """Index of the predecessor under the slot convention (-1 if none)."""
    prev = np.full(len(rid), -1)
    for i in range(1, len(rid)):
        j = i - 1
        while j >= 0 and rid[j] == rid[i] and pos[j] == pos[i]:
            j -= 1
        if j >= 0 and rid[j] == rid[i]:
            prev[i] = j
    return prev


def reply_lengths():
    lens = {}
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m in MODELS:
                out = r.get(f"{m} Output")
                if not pd.isna(out):
                    lens[f"{corpus}|{i}|{m}"] = len(norm(out))
    return lens


def synth(kind, rng, n_resp=800, n_ev=6):
    """Template (fixed thirds) or chain (repeat previous label w.p. 0.85)."""
    rows = []
    for r in range(n_resp):
        if kind == "template":
            k = max(1, n_ev // 3)
            evs = ([("E", rng.uniform(0, .33)) for _ in range(k)]
                   + [("A", rng.uniform(.34, .66)) for _ in range(k)]
                   + [("Q", rng.uniform(.67, 1)) for _ in range(k)])
        else:
            evs = sorted([(None, rng.uniform(0, 1)) for _ in range(n_ev)], key=lambda t: t[1])
            labs, prev = [], None
            for _ in evs:
                lab = prev if (prev and rng.random() < .85) else rng.choice(list("EAQ"))
                labs.append(lab)
                prev = lab
            evs = [(labs[i], evs[i][1]) for i in range(len(evs))]
        for lab, x in evs:
            rows.append(dict(response_id=r, label=lab, position=x))
    return pd.DataFrame(rows).sort_values(["response_id", "position"]).reset_index(drop=True)


# ============================================================ BINS
def stage_bins(E):
    print("\n=== BINS: bin count and binning scheme ===", flush=True)

    def compute_quantile(df, edges, n_shuffles=100, seed=0):
        b = np.clip(np.searchsorted(edges, df.position.to_numpy(), side="right") - 1, 0, len(edges) - 2)
        nb = len(edges) - 1
        d = df.copy()
        d["position"] = (b + 0.5) / nb
        return sm.compute(d, n_bins=nb, n_shuffles=n_shuffles, seed=seed)[0]

    rows = []
    for m in MODELS:
        d = sub(E, m)
        for B in [3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40]:
            res = sm.compute(d, n_bins=B, n_shuffles=100)[0]
            ce = sm.matched_ceiling(d, n_bins=B, n_shuffles=60)["SCRIPT"]
            rows.append(dict(model=m, B=B, scheme="fixed", SCRIPT=res["SCRIPT"], C=res["C_excess"],
                             M=res["M_excess"], z=res["z"], ceiling=ce, frac=round(res["SCRIPT"] / ce, 3)))
            q = np.quantile(d.position, np.linspace(0, 1, B + 1))
            q[0], q[-1] = 0.0, 1.0 + 1e-9
            rq = compute_quantile(d, q)
            rows.append(dict(model=m, B=B, scheme="quantile", SCRIPT=rq["SCRIPT"], C=rq["C_excess"],
                             M=rq["M_excess"], z=rq["z"], ceiling=np.nan, frac=np.nan))
            print(rows[-2], flush=True)
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / "q1_bin_sweep.csv", index=False)
    piv = Q[Q.scheme == "fixed"].pivot(index="B", columns="model", values="SCRIPT")
    print(piv.round(3))
    from scipy.stats import spearmanr
    ref = piv.loc[10]
    print("rank correlation with B=10 ordering:",
          {B: round(spearmanr(piv.loc[B], ref)[0], 2) for B in piv.index})
    print("argmax-z B per model:", Q[Q.scheme == "fixed"].pivot(index="B", columns="model", values="z").idxmax().to_dict())


# ============================================================ PERSON
def stage_person(L):
    print("\n=== PERSON: coverage, entropy, capacity-normalised shares ===", flush=True)
    meta = []
    for corpus in ["carebench", "hope"]:
        d = load_annotated(corpus)
        for i, r in d.iterrows():
            meta.append(dict(corpus=corpus, row=i, turn=int(r["Turn"]), **{u: int(r[u]) for u in UCOLS}))
    meta = pd.DataFrame(meta)
    cov = {u: meta[u].value_counts(normalize=True).round(3).to_dict() for u in UCOLS}
    print("coverage (pooled turns):", json.dumps(cov))
    HU = {u: _H(meta[u].value_counts().to_numpy().astype(float)) for u in UCOLS}
    joint = meta[UCOLS].astype(str).agg("|".join, axis=1)
    HU["joint"] = _H(joint.value_counts().to_numpy().astype(float))
    HU["n_joint_states"] = joint.nunique()
    print("H(U) bits:", {k: round(v, 3) for k, v in HU.items()})

    def share(Em, cols, n_perm=300, seed=0):
        U = Em[cols].astype(str).agg("|".join, axis=1)
        hl = _H(Em.label.value_counts().to_numpy().astype(float))
        hu = _H(U.value_counts().to_numpy().astype(float))
        obs = _mi(Em.label, U)
        rep = Em.groupby("response_id")[cols].first().astype(str).agg("|".join, axis=1)
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            perm = dict(zip(rep.index, rng.permutation(rep.to_numpy())))
            null.append(_mi(Em.label, Em.response_id.map(perm)))
        null = np.array(null)
        ex = obs - null.mean()
        return dict(share_L=round(ex / hl, 4), z=round(ex / null.std(), 2) if null.std() > 0 else np.nan,
                    share_U=round(ex / hu, 4), H_U=round(hu, 3), bound=round(hu / hl, 3))

    def seat_capacity(Em, n_shuf=100, seed=0):
        xb = np.minimum((Em.position.to_numpy() * 10).astype(int), 9)
        lab = pd.factorize(Em.label)[0]
        rid = pd.factorize(Em.response_id)[0]
        hl = _H(np.bincount(lab).astype(float))
        hx = _H(np.bincount(xb).astype(float))
        obs = _mi(lab, xb)
        rng = np.random.default_rng(seed)
        null = np.array([_mi(_shuffle_within(lab, rid, rng), xb) for _ in range(n_shuf)])
        ex = obs - null.mean()
        return dict(seat_L=round(ex / hl, 4), seat_X=round(ex / hx, 4), H_X=round(hx, 3))

    LM = L.merge(meta, on=["corpus", "row"], how="inner")
    rows = []
    for sp in MODELS + ["Human"]:
        Em = LM[LM.model == sp].sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
        r = dict(speaker=sp, n_events=len(Em), n_replies=Em.response_id.nunique())
        r.update({f"person_{k}": v for k, v in share(Em, UCOLS).items()})
        bal = share(Em, ["user_evocative", "user_typicality", "user_underlying"])
        r.update(person_balanced_share_L=bal["share_L"], person_balanced_z=bal["z"],
                 person_balanced_share_U=bal["share_U"])
        for u in UCOLS:
            s = share(Em, [u], n_perm=200)
            r[f"{u}_share_L"] = s["share_L"]
            r[f"{u}_z"] = s["z"]
            r[f"{u}_share_U"] = s["share_U"]
        r.update(seat_capacity(Em))
        rows.append(r)
        print(sp, {k: r[k] for k in ["person_share_L", "person_z", "person_share_U", "person_bound",
                                     "person_balanced_share_L", "seat_L", "seat_X"]}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q3_person_budget.csv", index=False)


# ============================================================ EXTENT
def stage_extent():
    print("\n=== EXTENT: start / midpoint / end, span length as a variable ===", flush=True)
    rows = []
    for corpus in CORPORA:
        df = load_annotated(corpus)
        for i, r in df.iterrows():
            for m, n in MODNUM.items():
                out = r.get(f"{m} Output")
                if pd.isna(out):
                    continue
                txt = norm(out)
                Ln = len(txt)
                if Ln == 0:
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
                        rows.append(dict(corpus=corpus, row=i, model=m, label=code, start=p / Ln,
                                         mid=(p + len(spn) / 2) / Ln,
                                         end=min((p + len(spn)) / Ln, 1.0 - 1e-9),
                                         length=len(spn), cover=len(spn) / Ln))
    X = pd.DataFrame(rows)
    X["response_id"] = X.corpus + "|" + X.row.astype(str) + "|" + X.model
    print("events with extent:", len(X), " median span length (chars):", X.length.median(),
          " median coverage:", round(X.cover.median(), 3))
    print("span length by code (median chars):",
          X.groupby("label").length.median().sort_values().round(0).to_dict())

    def length_term(d, n_shuf=100, seed=0):
        d = d.sort_values(["response_id", "start"], kind="stable").reset_index(drop=True)
        lab = pd.factorize(d.label)[0]
        rid = pd.factorize(d.response_id)[0]
        xb = np.minimum((d.start.to_numpy() * 10).astype(int), 9)
        dec = np.minimum((d.length.rank(pct=True).to_numpy() * 10).astype(int), 9)
        hl = _H(np.bincount(lab).astype(float))

        def cond(l):
            tot = 0.0
            for b in range(10):
                mk = xb == b
                if mk.sum() < 2:
                    continue
                tot += mk.mean() * _mi(l[mk], dec[mk])
            return tot / hl

        obs_c, obs_u = cond(lab), _mi(lab, dec) / hl
        rng = np.random.default_rng(seed)
        nc, nu = [], []
        for _ in range(n_shuf):
            s = _shuffle_within(lab, rid, rng)
            nc.append(cond(s))
            nu.append(_mi(s, dec) / hl)
        return dict(length_given_X=round(obs_c - np.mean(nc), 4),
                    length_given_X_z=round((obs_c - np.mean(nc)) / np.std(nc), 1),
                    length_uncond=round(obs_u - np.mean(nu), 4))

    rows = []
    for m in MODELS:
        d = X[X.model == m]
        r = dict(model=m)
        for pos in ["start", "mid", "end"]:
            dd = (d[["response_id", "label", pos]].rename(columns={pos: "position"})
                  .sort_values(["response_id", "position"], kind="stable").reset_index(drop=True))
            res = sm.compute(dd, n_shuffles=100)[0]
            r[f"SCRIPT_{pos}"] = res["SCRIPT"]
            r[f"C_{pos}"] = res["C_excess"]
            r[f"M_{pos}"] = res["M_excess"]
        r.update(length_term(d))
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q4_span_extent.csv", index=False)


# ============================================================ ORDER2
def order2(d, n_shuf=100, seed=0):
    """M1 and the order-2 gain I(L; L_prev2 | X, L_prev)/H(L), slot predecessors."""
    d = d.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
    lab = pd.factorize(d.label)[0]
    rid = pd.factorize(d.response_id)[0]
    pos = d.position.to_numpy()
    xb = np.minimum((pos * 10).astype(int), 9)
    hl = _H(np.bincount(lab).astype(float))
    prev = slot_predecessor(rid, pos)
    idx1 = np.flatnonzero(prev >= 0)
    idx2 = np.array([i for i in idx1 if prev[prev[i]] >= 0])
    p1 = prev[idx2]
    p2 = prev[p1]

    def terms(l):
        m1 = m2 = 0.0
        for b in range(10):
            mk = xb[idx1] == b
            if mk.sum() >= 2:
                m1 += mk.mean() * _mi(l[idx1][mk], l[prev[idx1]][mk])
            mk = xb[idx2] == b
            if mk.sum() >= 2:
                cur, a, bb = l[idx2][mk], l[p1[mk]], l[p2[mk]]
                m2 += mk.mean() * (_mi(cur, a * 100 + bb) - _mi(cur, a))
        return m1 / hl, m2 / hl

    o1, o2 = terms(lab)
    r = np.random.default_rng(seed)
    null = np.array([terms(_shuffle_within(lab, rid, r)) for _ in range(n_shuf)])
    return dict(M1=round(o1 - null[:, 0].mean(), 4), M2_gain=round(o2 - null[:, 1].mean(), 4),
                M2_z=round((o2 - null[:, 1].mean()) / null[:, 1].std(), 1), n_two_pred=len(idx2))


def stage_order2(E):
    print("\n=== ORDER2: order-2 gain at 5 groups and 20 codes ===", flush=True)
    GROUP = {**{c: "emp_acc" for c in ["VAC", "NAC", "ASAC", "SAC"]},
             **{c: "emp_in" for c in ["VIN", "NIN", "ASIN", "SIN"]},
             **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
             **{c: "quest" for c in ["QOP", "QCL", "TEN"]}}
    rows = []
    for m in MODELS:
        g = sub(E, m)
        g["label"] = g.label.map(GROUP).fillna("other")
        r = dict(model=m, level="5-group", **order2(g))
        rows.append(r)
        print(r, flush=True)
    for m in MODELS:
        r = dict(model=m, level="20-code", **order2(sub(E, m)))
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q5_order2.csv", index=False)


# ============================================================ DENSITY
def stage_density(E):
    print("\n=== DENSITY: synthetic densities; reply-length terciles with matched ceiling ===", flush=True)
    rng = np.random.default_rng(7)
    rows = []
    for kind in ["template", "chain"]:
        for n_ev in [3, 6, 12, 24]:
            for n_resp in [200, 800]:
                d = synth(kind, rng, n_resp=n_resp, n_ev=n_ev)
                res = sm.compute(d, n_shuffles=60)[0]
                ce = sm.matched_ceiling(d, n_shuffles=60)["SCRIPT"] if kind == "chain" else np.nan
                rows.append(dict(kind=kind, n_resp=n_resp, events_per_reply=n_ev, n_events=res["n_events"],
                                 SCRIPT=res["SCRIPT"], C=res["C_excess"], M=res["M_excess"], z=res["z"],
                                 ceiling=ce, fraction=round(res["SCRIPT"] / ce, 3) if kind == "chain" else np.nan))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q6_synthetic_density.csv", index=False)

    lens = reply_lengths()
    E = E.copy()
    E["reply_len"] = E.response_id.map(lens)
    rows, fr = [], []
    for m in MODELS:
        d = E[E.model == m]
        rl = d.groupby("response_id").reply_len.first()
        ter = pd.qcut(rl, 3, labels=["short", "medium", "long"])
        for t in ["short", "medium", "long"]:
            ids = ter[ter == t].index
            dd = d[d.response_id.isin(ids)][["response_id", "label", "position"]].reset_index(drop=True)
            res = sm.compute(dd, n_shuffles=100)[0]
            rows.append(dict(model=m, stratum=t, n_replies=len(ids), median_len=int(rl[ids].median()),
                             events_per_reply=round(len(dd) / len(ids), 1), SCRIPT=res["SCRIPT"],
                             C=res["C_excess"], M=res["M_excess"], z=res["z"]))
            ce = sm.matched_ceiling(dd, n_shuffles=60)["SCRIPT"]
            fr.append(dict(model=m, stratum=t, SCRIPT=res["SCRIPT"], ceiling=round(ce, 4),
                           fraction=round(res["SCRIPT"] / ce, 3), C=res["C_excess"]))
            print(rows[-1], fr[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q6_reply_length.csv", index=False)
    pd.DataFrame(fr).to_csv(OUT / "q6_fraction_by_length.csv", index=False)


# ============================================================ SEQPOWER
def stage_seqpower(E):
    print("\n=== SEQPOWER: SCRIPT-Seq stopping times, real and synthetic ===", flush=True)
    from scriptmetric import betting as sb
    TGRID = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300]

    def summarise(taus, **kw):
        taus = np.array(taus, float)
        fin = taus[np.isfinite(taus)]
        row = dict(**kw, streams=len(taus), detected=int(np.isfinite(taus).sum()),
                   median_tau=float(np.median(fin)) if len(fin) else np.nan,
                   q90_tau=float(np.quantile(fin, .9)) if len(fin) else np.nan,
                   max_tau=float(fin.max()) if len(fin) else np.nan)
        for t in TGRID:
            row[f"power_t{t}"] = round(float((taus <= t).mean()), 3)
        return row

    rows = []
    for m in MODELS:
        d = sub(E, m)
        for alpha in [0.05, 0.01]:
            taus = []
            for s in range(30):
                r = sb.test_structure(d, alpha=alpha, seed=s)
                taus.append(r["tau"] if r["rejected"] else np.inf)
            rows.append(summarise(taus, system=m, alpha=alpha))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q7_real_power.csv", index=False)

    d = sub(E, "Claude")
    rows = []
    for kept in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:
        for alpha in [0.05, 0.01]:
            taus = []
            for s in range(30):
                r_ = np.random.default_rng(1000 + s)
                ids = d.response_id.unique()
                shuf_ids = set(r_.choice(ids, int(round((1 - kept) * len(ids))), replace=False))
                parts = []
                for rid, g in d.groupby("response_id", sort=False):
                    g = g.copy()
                    if rid in shuf_ids:
                        g["label"] = r_.permutation(g.label.to_numpy())
                    parts.append(g)
                dd = pd.concat(parts, ignore_index=True)
                res = sb.test_structure(dd, alpha=alpha, seed=s)
                taus.append(res["tau"] if res["rejected"] else np.inf)
            rows.append(summarise(taus, signal_kept=kept, alpha=alpha))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "q7_synthetic_power.csv", index=False)


# ============================================================ TAGGER
def stage_tagger():
    print("\n=== TAGGER: position dependence of the Testbed-2 tagger ===", flush=True)
    from pipeline.external.empathy_checks import sent_tokenize
    root = MINT_DIR / "evaluation" / "outputs"
    rows = []
    for d in sorted(root.iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        for e in json.load(open(f)):
            text = e["model_response"].strip()
            sents = sent_tokenize(text) if text else []
            st = e["sentence_tactics"]
            if isinstance(st, str):
                st = eval(st)
            if len(sents) != len(st) or not sents:
                continue
            n = len(sents)
            for i, (s, tacs) in enumerate(zip(sents, st)):
                rows.append(dict(system=d.name, human=d.name == "gold", rank=i / max(n - 1, 1), n_sents=n,
                                 is_q=s.strip().endswith("?"), tag_q="questioning" in tacs))
    S = pd.DataFrame(rows)
    S["pos"] = pd.cut(S["rank"], [-.01, .2, .4, .6, .8, 1.0], labels=["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])
    print("sentences:", len(S), " systems:", S.system.nunique())
    for lab, grp in [("models", S[~S.human]), ("human gold", S[S.human])]:
        t = grp.groupby("pos", observed=True).agg(
            n=("is_q", "size"), share_lexical_q=("is_q", "mean"), share_tag_q=("tag_q", "mean"),
            P_tag_given_qmark=("tag_q", lambda x: x[grp.loc[x.index, "is_q"]].mean()),
            P_tag_given_noqmark=("tag_q", lambda x: x[~grp.loc[x.index, "is_q"]].mean()))
        print(f"\n{lab}:\n", t.round(3).to_string())
        t.assign(layer=lab).to_csv(OUT / f"q2_tagger_position_{lab.replace(' ', '_')}.csv")

    # position-blind rule labeller on the same sentences
    MODAL = re.compile(r"\b(you could|you might|you can|try|consider|maybe|perhaps|it might help|it may help|"
                       r"i('d| would) (suggest|recommend)|have you (tried|considered))\b")
    EMO = re.compile(r"\b(i('m| am) (so )?sorry|that sounds|it sounds|i can (only )?imagine|that must|"
                     r"it('s| is) (completely |totally )?(understandable|normal|okay|valid)|"
                     r"you('re| are) not alone|i hear you)\b")

    def lex(s):
        s = s.strip().lower()
        if s.endswith("?"):
            return "question"
        if MODAL.search(s):
            return "advice"
        if EMO.search(s):
            return "empathy"
        return "other"

    rows = []
    for d in sorted(root.iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        for k, e in enumerate(json.load(open(f))):
            text = e["model_response"].strip()
            sents = sent_tokenize(text) if text else []
            if not sents:
                continue
            cur = 0
            for s in sents:
                p = text.find(s, cur)
                if p < 0:
                    continue
                cur = p + len(s)
                rows.append(dict(system=d.name, response_id=f"{e['conversation_id']}|{k}", label=lex(s),
                                 position=min(p / len(text), 1 - 1e-9)))
    S = pd.DataFrame(rows)
    tagger = pd.read_csv(TAB / "mint_systems.csv").set_index("system")
    res = []
    for sysname, g in S.groupby("system"):
        g = g.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
        r = sm.compute(g[["response_id", "label", "position"]], n_shuffles=200)[0]
        res.append(dict(system=sysname, SCRIPT_lex=r["SCRIPT"], C_lex=r["C_excess"], M_lex=r["M_excess"],
                        z_lex=r["z"], n_events=r["n_events"],
                        SCRIPT_tagger=tagger.loc[sysname, "SCRIPT"], C_tagger=tagger.loc[sysname, "C"],
                        share_other=round((g.label == "other").mean(), 3)))
        print(res[-1], flush=True)
    R = pd.DataFrame(res).sort_values("SCRIPT_lex")
    R.to_csv(OUT / "q2_lexical_labeller.csv", index=False)
    from scipy.stats import spearmanr
    m = R[R.system != "gold"]
    print("Spearman(system SCRIPT, lexical vs tagger) over 22 systems:",
          round(spearmanr(m.SCRIPT_lex, m.SCRIPT_tagger)[0], 3))
    print("human gold lexical SCRIPT:", R[R.system == "gold"].SCRIPT_lex.item(),
          " models range:", m.SCRIPT_lex.min(), "-", m.SCRIPT_lex.max())


# ============================================================ PROFILES
def stage_profiles():
    print("\n=== PROFILES: what moves a profile (Testbed 2) ===", flush=True)
    from pipeline.external.empathy_checks import turn_events
    root = MINT_DIR / "evaluation" / "outputs"
    profs, meta8, evs = {}, {}, {}
    for d in sorted(root.iterdir()):
        f = d / "conversations_tagged.json"
        if not f.exists():
            continue
        Ev, _ = turn_events(json.load(open(f)), d.name)
        Ev = Ev.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
        res, prof = sm.compute(Ev[["response_id", "label", "position"]], n_shuffles=60)
        profs[d.name] = prof
        evs[d.name] = Ev
        key = re.sub(r"_Qwen3-(1\.7|4)B$", "", d.name)
        size = re.search(r"Qwen3-(1\.7B|4B)", d.name)
        meta8[d.name] = dict(recipe=key, size=size.group(1) if size else "human", SCRIPT=res["SCRIPT"],
                             C=res["C_excess"], M=res["M_excess"])
    names = [n for n in profs if meta8[n]["size"] != "human"]
    D = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            D.loc[a, b] = sm.profile_distance(profs[a], profs[b])
    D.round(4).to_csv(OUT / "q8_js_matrix.csv")
    same_recipe, same_size, neither = [], [], []
    for a in names:
        for b in names:
            if a >= b:
                continue
            ma, mb = meta8[a], meta8[b]
            v = D.loc[a, b]
            (same_recipe if ma["recipe"] == mb["recipe"] else same_size if ma["size"] == mb["size"]
             else neither).append(v)
    summary = dict(js_same_recipe_other_size=round(float(np.mean(same_recipe)), 4),
                   js_same_size_other_recipe=round(float(np.mean(same_size)), 4),
                   js_neither=round(float(np.mean(neither)), 4))
    print(summary)
    for size in ["1.7B", "4B"]:
        base = f"baseline1_vanilla_Qwen3-{size}"
        drift = {meta8[n]["recipe"]: round(D.loc[base, n], 3) for n in names if meta8[n]["size"] == size and n != base}
        other = f"baseline1_vanilla_Qwen3-{'4B' if size == '1.7B' else '1.7B'}"
        print(size, "JS vanilla -> intervention:", drift, "| vanilla -> vanilla other size:", round(D.loc[base, other], 3))
        summary[f"js_vanilla_{size}_to_other_size"] = round(float(D.loc[base, other]), 4)
        for k, v in drift.items():
            summary[f"js_vanilla_{size}_to_{k}"] = v
    rng = np.random.default_rng(0)
    hit_self = hit_recipe = hit_size = n = 0
    for a in names:
        Ev = evs[a]
        ids = Ev.response_id.unique()
        rng.shuffle(ids)
        probe_ids = set(ids[:40])
        probe = Ev[Ev.response_id.isin(probe_ids)][["response_id", "label", "position"]]
        refs = {}
        for b in names:
            Eb = evs[b]
            Eb = Eb[~Eb.response_id.isin(probe_ids)] if b == a else Eb
            _, pr = sm.compute(Eb[["response_id", "label", "position"]], n_shuffles=1)
            refs[b] = pr
        scores = {b: sm.profile_loglik(probe, refs[b]) for b in names}
        best = max(scores, key=scores.get)
        n += 1
        hit_self += best == a
        hit_recipe += meta8[best]["recipe"] == meta8[a]["recipe"]
        hit_size += meta8[best]["size"] == meta8[a]["size"]
    print(f"40-turn probe over {n} systems: self {hit_self}/{n}, same recipe {hit_recipe}/{n}, "
          f"same size {hit_size}/{n} (chance self 1/{n}, recipe 2/{n}, size {n // 2}/{n})")
    summary.update(probe_self=hit_self, probe_recipe=hit_recipe, probe_size=hit_size, probe_n=n)
    M8 = pd.DataFrame(meta8).T
    M8.to_csv(OUT / "q8_systems.csv")
    json.dump(summary, open(OUT / "q8_summary.json", "w"), indent=1)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    stages = ["BINS", "PERSON", "EXTENT", "ORDER2", "DENSITY", "SEQPOWER", "TAGGER", "PROFILES"] \
        if stage == "ALL" else [stage]
    E = load_events(EVENTS)
    L = load_events(DERIVED / "llm_span_events.csv") if (DERIVED / "llm_span_events.csv").exists() else None
    for s in stages:
        if s == "BINS":
            stage_bins(E)
        elif s == "PERSON":
            stage_person(L)
        elif s == "EXTENT":
            stage_extent()
        elif s == "ORDER2":
            stage_order2(E)
        elif s == "DENSITY":
            stage_density(E)
        elif s == "SEQPOWER":
            stage_seqpower(E)
        elif s == "TAGGER":
            stage_tagger()
        elif s == "PROFILES":
            stage_profiles()
        else:
            sys.exit(f"unknown stage {s}")
    print("done")


if __name__ == "__main__":
    main()
