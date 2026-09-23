#!/usr/bin/env python3
"""SCRIPT lifted to conversation scale on the two multi-turn corpora.

Conversation choreography I(L; turn)/H(L) asks whether the turn index schedules the
behaviour mix; boundary momentum I(first of turn t+1; last of turn t)/H asks whether a
reply's opening remembers the previous reply's closing. Both are calibrated against
within-conversation permutation nulls, at the 5-group and 20-code level.

    python -m pipeline.metric.multiturn [carebench|hope]      (default: both, plus pooled)

Writes tables/multiturn_extension.csv and multiturn_boundary_lift.csv."""
import numpy as np
import pandas as pd

from pipeline.common.paths import EVENTS, TAB

RNG_SEED = 0
N_SHUF = 1000

GROUPS = {"emp_acc": ["VAC", "NAC", "ASAC", "SAC"],
          "emp_in": ["VIN", "NIN", "ASIN", "SIN"],
          "advice": ["DIR", "FIX", "RECT"],
          "quest": ["QOP", "QCL"],
          "other": ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}
C2G = {c: g for g, cs in GROUPS.items() for c in cs}

MODELS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
MT_CORPORA = ["carebench", "hope"]

DATA_DIR_OVERRIDE = None  # optional CLI override of the raw-data directory


def _H(counts):
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _mi(a, b, na, nb):
    j = np.zeros((na, nb))
    np.add.at(j, (a, b), 1.0)
    return _H(j.sum(1)) + _H(j.sum(0)) - _H(j.ravel())


def load_events_with_turns(data_dir=None):
    """Attach (conversation, turn) to the multi-turn events.

    `row` in span_events.csv is the raw CSV row index, so Conversation/Turn
    come straight from the raw annotated CSVs (DATA_DIR, i.e. the HF download).
    Falls back to file-order mapping via a spans_long.csv path if a raw CSV
    is unavailable.
    """
    from pipeline.common.paths import DATA_DIR
    data_dir = data_dir or DATA_DIR
    ev = pd.read_csv(EVENTS)
    frames = []
    for ds in MT_CORPORA:
        raw = pd.read_csv(f"{data_dir}/{ds}_annotated.csv", low_memory=False,
                          usecols=["Conversation", "Turn"])
        e = ev[ev.corpus == ds].copy()
        e["conv"] = e.row.map(raw.Conversation.astype(str).to_dict())
        e["turn"] = e.row.map(raw.Turn.astype(int).to_dict())
        frames.append(e)
    E = pd.concat(frames, ignore_index=True)
    return E.sort_values(["corpus", "model", "conv", "turn", "position"]).reset_index(drop=True)


def conversation_choreography(e, lab_col, n_shuf=N_SHUF, seed=RNG_SEED):
    """C_x = I(L; turn)/H(L), null = within-conversation label permutation."""
    labs = sorted(e[lab_col].unique())
    l2i = {l: i for i, l in enumerate(labs)}
    lab = e[lab_col].map(l2i).to_numpy()
    turn = e.turn.to_numpy() - 1
    n_turn = int(turn.max()) + 1
    cid = pd.factorize(e.conv)[0]
    H_L = _H(np.bincount(lab, minlength=len(labs)).astype(float))
    if H_L == 0:
        return dict(Cx=np.nan, z=np.nan, raw=np.nan, null=np.nan, n=len(e))
    obs = _mi(lab, turn, len(labs), n_turn) / H_L
    rng = np.random.default_rng(seed)
    segs = [np.flatnonzero(cid == c) for c in range(cid.max() + 1)]
    null = np.empty(n_shuf)
    shuf = lab.copy()
    for s in range(n_shuf):
        for seg in segs:
            shuf[seg] = lab[seg][rng.permutation(len(seg))]
        null[s] = _mi(shuf, turn, len(labs), n_turn) / H_L
    return dict(Cx=obs - null.mean(),
                z=(obs - null.mean()) / null.std() if null.std() > 0 else np.nan,
                raw=obs, null=null.mean(), n=len(e))


def boundary_pairs(e, lab_col):
    """(last label of turn t, first label of turn t+1) per conversation."""
    first = e.groupby(["conv", "turn"]).first()[lab_col]
    last = e.groupby(["conv", "turn"]).last()[lab_col]
    P, F, conv_of = [], [], []
    for (cv, t) in last.index:
        if (cv, t + 1) in first.index:
            P.append(last.loc[(cv, t)])
            F.append(first.loc[(cv, t + 1)])
            conv_of.append(cv)
    return np.array(P), np.array(F), np.array(conv_of)


def boundary_momentum(e, lab_col, n_shuf=N_SHUF, seed=RNG_SEED):
    """M_x = I(F; P)/H(F), null = permute F across boundaries within conv."""
    P, F, conv = boundary_pairs(e, lab_col)
    labs = sorted(set(P) | set(F))
    l2i = {l: i for i, l in enumerate(labs)}
    p = np.array([l2i[x] for x in P])
    f = np.array([l2i[x] for x in F])
    cid = pd.factorize(conv)[0]
    H_F = _H(np.bincount(f, minlength=len(labs)).astype(float))
    if H_F == 0 or len(f) < 10:
        return dict(Mx=np.nan, z=np.nan, raw=np.nan, null=np.nan, n=len(f))
    obs = _mi(f, p, len(labs), len(labs)) / H_F
    rng = np.random.default_rng(seed)
    segs = [np.flatnonzero(cid == c) for c in range(cid.max() + 1)]
    null = np.empty(n_shuf)
    shuf = f.copy()
    for s in range(n_shuf):
        for seg in segs:
            shuf[seg] = f[seg][rng.permutation(len(seg))]
        null[s] = _mi(shuf, p, len(labs), len(labs)) / H_F
    return dict(Mx=obs - null.mean(),
                z=(obs - null.mean()) / null.std() if null.std() > 0 else np.nan,
                raw=obs, null=null.mean(), n=len(f))


def boundary_lift(e, lab_col):
    """Observed / expected boundary transitions (for the qualitative table)."""
    P, F, _ = boundary_pairs(e, lab_col)
    labs = sorted(set(P) | set(F))
    rows = []
    pF = pd.Series(F).value_counts(normalize=True)
    for a in labs:
        sel = P == a
        if sel.sum() < 8:
            continue
        for b in labs:
            obs = (F[sel] == b).mean()
            exp = pF.get(b, 0)
            if exp > 0:
                rows.append(dict(prev_last=a, next_first=b, n_prev=int(sel.sum()),
                                 p_obs=round(obs, 3), p_marginal=round(exp, 3),
                                 lift=round(obs / exp, 2)))
    return pd.DataFrame(rows)


def main():
    e_all = load_events_with_turns(DATA_DIR_OVERRIDE)
    e_all["group"] = e_all.label.map(C2G)

    rows = []
    for lab_col, gran in [("group", "5-group"), ("label", "20-code")]:
        for corpus in MT_CORPORA + ["pooled"]:
            for model in MODELS:
                e = e_all if corpus == "pooled" else e_all[e_all.corpus == corpus]
                e = e[e.model == model]
                if corpus == "pooled":
                    e = e.assign(conv=e.corpus + "|" + e.conv.astype(str))
                cx = conversation_choreography(e, lab_col)
                mx = boundary_momentum(e, lab_col)
                rows.append(dict(granularity=gran, corpus=corpus, model=model,
                                 Cx=round(cx["Cx"], 4), z_Cx=round(cx["z"], 1),
                                 Cx_raw=round(cx["raw"], 4), Cx_null=round(cx["null"], 4),
                                 n_events=cx["n"],
                                 Mx=round(mx["Mx"], 4), z_Mx=round(mx["z"], 1),
                                 Mx_raw=round(mx["raw"], 4), Mx_null=round(mx["null"], 4),
                                 n_boundaries=mx["n"]))
                print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "multiturn_extension.csv", index=False)

    lifts = []
    for model in MODELS:
        e = e_all[e_all.model == model].assign(
            conv=lambda d: d.corpus + "|" + d.conv.astype(str))
        lf = boundary_lift(e, "group")
        lf.insert(0, "model", model)
        lifts.append(lf)
    pd.concat(lifts).to_csv(TAB / "multiturn_boundary_lift.csv", index=False)
    print(f"\nwrote {TAB / 'multiturn_extension.csv'}")
    print(f"wrote {TAB / 'multiturn_boundary_lift.csv'}")


if __name__ == "__main__":
    import sys
    DATA_DIR_OVERRIDE = sys.argv[1] if len(sys.argv) > 1 else None
    main()
