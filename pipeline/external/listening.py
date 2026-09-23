#!/usr/bin/env python3
"""Does the blueprint listen? Therapist and models on one annotation layer.

On the multi-turn corpora, with every speaker annotated by the same blind LLM
(external/llm_annotator.py --include-therapist), each in excess of a permutation null
that re-deals whole replies:
  arc / drift     behaviour mix by turn; JS distance between opening and closing turns
  user state      JS distance between the mixes of two coded user states
  budget          share of H(L) explained by the seat (C), the person and the turn
  level by turn   SCRIPT inside each turn bucket, both annotation layers
  seat strength   per-code positional information on the clinician layer

Writes tables/listening_*.csv, script_by_turn.csv, seat_strength.csv."""
import sys

import numpy as np
import pandas as pd

from pipeline.common.paths import CODES, DATA_DIR, DERIVED, EVENTS, MODELS, TAB

from scriptmetric import metric as sm

N_PERM, N_SHUFFLES, SEED = 300, 200, 0
GROUP = {**{c: "emp_acc" for c in ["VAC", "NAC", "ASAC", "SAC"]},
         **{c: "emp_in" for c in ["VIN", "NIN", "ASIN", "SIN"]},
         **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
         **{c: "quest" for c in ["QOP", "QCL", "TEN"]}}
G5 = ["emp_acc", "emp_in", "advice", "quest", "other"]
SPEAKERS = MODELS + ["Human"]
BUCKETS = ["T1-3", "T4-7", "T8-10"]
CUES = [("user_sensitivity", 0, 1, "safety cue"),
        ("user_evocative", 0, 2, "affective intensity"),
        ("user_typicality", 0, 2, "atypical presentation"),
        ("user_underlying", 1, 2, "implicit distress"),
        ("user_request_info", 0, 1, "explicit request")]
UCOLS = [c for c, *_ in CUES]


# ------------------------------------------------------------------- loading

def turn_meta():
    rows = []
    for corpus in ["carebench", "hope"]:
        d = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        d.columns = [c.lstrip("\ufeff").strip() for c in d.columns]
        for i, r in d.iterrows():
            rows.append(dict(corpus=corpus, row=i, turn=int(r["Turn"]),
                             **{u: int(r[u]) for u in UCOLS}))
    return pd.DataFrame(rows)


def prep(E, meta):
    E = E.merge(meta, on=["corpus", "row"], how="inner").copy()
    E["g"] = E.label.map(GROUP).fillna("other")
    E["rid"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    E["tb"] = pd.cut(E.turn, [0, 3, 7, 10], labels=BUCKETS).astype(str)
    return E.sort_values(["rid", "position"]).reset_index(drop=True)


# --------------------------------------------------------------------- stats

def _H(c):
    c = np.asarray(c, float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def _mi(a, b):
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    return _H(ct.sum(1)) + _H(ct.sum(0)) - _H(ct.ravel())


def js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def mix(E, key, cats):
    return E[key].value_counts().reindex(cats).fillna(0).to_numpy()


def strata_js(E, col, a, b, key, cats, seed=SEED):
    """JS between behaviour mixes in stratum a vs b, minus a null that re-deals
    whole replies to strata (same stratum sizes, reply integrity kept)."""
    E = E[E[col].astype(str).isin([str(a), str(b)])]
    rep = E.groupby("rid")[col].first().astype(str)
    obs = js(mix(E[E[col].astype(str) == str(a)], key, cats),
             mix(E[E[col].astype(str) == str(b)], key, cats))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_PERM):
        perm = dict(zip(rep.index, rng.permutation(rep.to_numpy())))
        s = E.rid.map(perm)
        null.append(js(mix(E[s == str(a)], key, cats), mix(E[s == str(b)], key, cats)))
    null = np.array(null)
    return dict(JS=round(obs, 4), excess=round(obs - null.mean(), 4),
                z=round((obs - null.mean()) / null.std(), 2) if null.std() > 0 else np.nan,
                n_events=int(len(E)), n_replies_a=int((rep == str(a)).sum()),
                n_replies_b=int((rep == str(b)).sum()))


def explained_share(E, cols, seed=SEED):
    """I(L;U)/H(L) for U = joint state over cols, minus the null that re-deals
    whole replies across states."""
    U = E[cols].astype(str).agg("|".join, axis=1)
    hl = _H(E.label.value_counts().to_numpy())
    obs = _mi(E.label, U) / hl
    rep = E.groupby("rid")[cols].first().astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_PERM):
        perm = dict(zip(rep.index, rng.permutation(rep.to_numpy())))
        null.append(_mi(E.label, E.rid.map(perm)) / hl)
    null = np.array(null)
    return round(obs - null.mean(), 4), round((obs - null.mean()) / null.std(), 2)


def seat_strength(E, n_perm=100, seed=SEED):
    """Per-label KL(P(X|l) || P(X)) over 10 position bins, minus the within-
    reply shuffle null: how tightly each behaviour is seated."""
    xb = np.minimum((E.position.to_numpy() * 10).astype(int), 9)
    E = E.assign(xb=xb)
    px = np.bincount(xb, minlength=10) / len(E)

    def per_label(lab):
        out = {}
        for l, g in E.assign(label=lab.to_numpy()).groupby("label"):
            p = np.bincount(g.xb, minlength=10) / len(g)
            m = p > 0
            out[l] = float((p[m] * np.log2(p[m] / px[m])).sum())
        return pd.Series(out)

    obs = per_label(E.label)
    rng = np.random.default_rng(seed)
    null = pd.concat([per_label(E.groupby("rid")["label"]
                                .transform(lambda s: rng.permutation(s.to_numpy())))
                      for _ in range(n_perm)], axis=1)
    S = pd.DataFrame(dict(KL_bits=obs.round(4), excess=(obs - null.mean(1)).round(4),
                          z=((obs - null.mean(1)) / null.std(1)).round(2),
                          n_events=E.label.value_counts()))
    S["group"] = S.index.map(GROUP).fillna("other")
    S["mean_position"] = E.groupby("label").position.mean().round(3)
    return S.sort_values("excess", ascending=False)


# ---------------------------------------------------------------------- main

def main():
    meta = turn_meta()
    llm_path = DERIVED / "llm_span_events.csv"
    if not llm_path.exists():
        sys.exit("run pipeline.external.llm_annotator --include-therapist first")
    L = prep(pd.read_csv(llm_path), meta)          # DeepSeek layer, six speakers
    C = prep(pd.read_csv(EVENTS), meta)            # clinician layer, five models
    print("DeepSeek-layer events (multi-turn):", L.groupby("model").size().to_dict())

    # --- arc: mix by turn (1..10) and by bucket
    arc = []
    for sp in SPEAKERS:
        E = L[L.model == sp]
        for key, cats in [("g", G5), ("label", CODES)]:
            for t, g in E.groupby("turn"):
                sh = mix(g, key, cats) / len(g) * 100
                for c, v in zip(cats, sh):
                    arc.append(dict(speaker=sp, alphabet=key, turn=int(t), bucket=None,
                                    category=c, share_pct=round(v, 2), n_events=len(g)))
            for tb, g in E.groupby("tb"):
                sh = mix(g, key, cats) / len(g) * 100
                for c, v in zip(cats, sh):
                    arc.append(dict(speaker=sp, alphabet=key, turn=None, bucket=tb,
                                    category=c, share_pct=round(v, 2), n_events=len(g)))
    pd.DataFrame(arc).to_csv(TAB / "listening_arc.csv", index=False)

    # --- drift across the conversation
    drift = []
    for sp in SPEAKERS:
        E = L[L.model == sp]
        for key, cats in [("g", G5), ("label", CODES)]:
            r = strata_js(E, "tb", "T1-3", "T8-10", key, cats)
            drift.append(dict(speaker=sp, alphabet="5-group" if key == "g" else "20-code", **r))
    D = pd.DataFrame(drift)
    D.to_csv(TAB / "listening_drift.csv", index=False)

    # --- user state
    us = []
    for sp in SPEAKERS:
        E = L[L.model == sp]
        for col, a, b, name in CUES:
            for key, cats in [("g", G5), ("label", CODES)]:
                r = strata_js(E, col, a, b, key, cats)
                us.append(dict(speaker=sp, cue=name, column=col, low=a, high=b,
                               alphabet="5-group" if key == "g" else "20-code", **r))
    U = pd.DataFrame(us)
    U.to_csv(TAB / "listening_user_state.csv", index=False)

    # --- mix by affective intensity (for the figure)
    aff = []
    for sp in SPEAKERS:
        E = L[L.model == sp]
        for lvl, g in E.groupby("user_evocative"):
            sh = mix(g, "g", G5) / len(g) * 100
            for c, v in zip(G5, sh):
                aff.append(dict(speaker=sp, user_evocative=int(lvl), category=c,
                                share_pct=round(v, 2), n_events=len(g)))
    pd.DataFrame(aff).to_csv(TAB / "listening_affect_mix.csv", index=False)

    # --- information budget
    bud = []
    for sp in SPEAKERS:
        E = L[L.model == sp].rename(columns={"rid": "response_id"})
        res, _ = sm.compute(E[["response_id", "label", "position"]], n_shuffles=N_SHUFFLES, seed=SEED)
        E = E.rename(columns={"response_id": "rid"})
        pu, zu = explained_share(E, UCOLS)
        pt, zt = explained_share(E, ["tb"])
        bud.append(dict(speaker=sp, seat_share=res["C_excess"], seat_z=res["z"],
                        person_share=pu, person_z=zu, turn_share=pt, turn_z=zt,
                        seat_over_person=round(res["C_excess"] / pu, 1) if pu > 0 else np.nan,
                        SCRIPT=res["SCRIPT"], n_events=res["n_events"],
                        n_replies=res["n_responses"],
                        events_per_reply=round(res["n_events"] / res["n_responses"], 2)))
    B = pd.DataFrame(bud)
    B.to_csv(TAB / "listening_budget.csv", index=False)

    # --- SCRIPT level by turn bucket, both layers
    lev = []
    for layer, E, spk in [("LLM annotator", L, SPEAKERS), ("clinicians", C, MODELS)]:
        for sp in spk:
            for tb in BUCKETS:
                d = E[(E.model == sp) & (E.tb == tb)].rename(columns={"rid": "response_id"})
                r, _ = sm.compute(d[["response_id", "label", "position"]], n_shuffles=N_SHUFFLES, seed=SEED)
                lev.append(dict(layer=layer, speaker=sp, bucket=tb, SCRIPT=r["SCRIPT"], z=r["z"],
                                C=r["C_excess"], M=r["M_excess"], n_events=r["n_events"],
                                n_replies=r["n_responses"]))
    pd.DataFrame(lev).to_csv(TAB / "script_by_turn.csv", index=False)

    # --- seat strength per label (clinician layer, all four corpora)
    Call = pd.read_csv(EVENTS)
    Call["rid"] = Call.corpus + "|" + Call.row.astype(str) + "|" + Call.model
    S = seat_strength(Call)
    S.to_csv(TAB / "seat_strength.csv")

    pd.set_option("display.width", 220)
    print("\n-- drift T1-3 vs T8-10 (5-group)\n", D[D.alphabet == "5-group"].to_string(index=False))
    print("\n-- user state (5-group), excess JS\n",
          U[U.alphabet == "5-group"].pivot(index="speaker", columns="cue", values="excess").loc[SPEAKERS])
    print("\n-- user state (5-group), z\n",
          U[U.alphabet == "5-group"].pivot(index="speaker", columns="cue", values="z").loc[SPEAKERS])
    print("\n-- information budget\n", B.to_string(index=False))
    print("\n-- seat strength (clinician layer)\n", S.to_string())
    print("\nwrote listening_*.csv, script_by_turn.csv, seat_strength.csv")


if __name__ == "__main__":
    main()
