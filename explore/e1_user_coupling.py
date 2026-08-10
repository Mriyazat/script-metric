#!/usr/bin/env python3
"""Does the script actually ignore the user? (exploratory — not in the paper)

SCRIPT measures how much of a model's behaviour is explained by *response
structure*. It never looks at the user's message, which leaves the obvious
question unanswered: maybe the user's message would explain just as much, and
the structure is only a proxy for content that happens to be arranged.

Two tests, both using only data already in the benchmark.

TEST 1 — responsiveness to the user's clinically-rated state
------------------------------------------------------------
Each item carries clinician ratings of the *user's* turn: whether they
explicitly ask for information, how emotionally evocative the turn is, whether
it is sensitive/risky, whether there is an underlying issue, how typical it is,
plus the topic. These instantiate exactly the contrast of interest — the same
broad situation, but a different thing a good reply would have to do.

    A_u = I(L ; u | X) / H(L)          responsiveness to user state u

calibrated against a null that permutes u across responses within corpus,
holding every label sequence, every position and every marginal fixed. That is
the mirror image of SCRIPT's own null: SCRIPT breaks the label-position link,
this breaks the user-state-behaviour link. Same scale, so A_u and C are
directly comparable numbers.

TEST 2 — matched head-to-head: user's message vs model identity
---------------------------------------------------------------
Every prompt is answered by all five models, so for any reply there are two
equally-sized reference sets of four other replies:

    same prompt, other models   (the user's message held fixed)
    same model, other prompts   (the model held fixed)

Build a profile from each and ask which better predicts the held-out reply's
arrangement, by mean per-event log-likelihood. Exactly four replies either
side, so this is matched by construction — no cardinality or bias asymmetry.

If arrangement were driven by what the user said, the same-prompt reference
should win. Run:

    python e1_user_coupling.py [--shuffles 200] [--draws 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from paths import CORPORA, DATA_DIR, EVENTS, MODELS, OUT   # noqa: E402

import script_metric as sm   # noqa: E402

N_BINS = 10
OUTDIR = OUT / "explore"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Clinician ratings of the *user's* turn. Each is a small ordinal/binary scale.
USER_VARS = {
    "user_request_info": "user explicitly asks for information/advice",
    "user_evocative": "how emotionally evocative the user's turn is",
    "user_sensitivity": "sensitive / risk-bearing content",
    "user_underlying": "an underlying issue the user has not stated",
    "user_typicality": "how typical the presentation is",
    "Mental Health Topic": "topic of the user's turn",
}


# ------------------------------------------------------------------ test 1

def _H(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _cond_mi(lab, u, xbin, n_lab, n_u, n_bins) -> float:
    """I(L ; U | X), averaged over bins with weights p(x)."""
    total, n = 0.0, len(lab)
    for b in range(n_bins):
        m = xbin == b
        k = int(m.sum())
        if k < 2:
            continue
        j = np.zeros((n_lab, n_u))
        np.add.at(j, (lab[m], u[m]), 1.0)
        total += (k / n) * (_H(j.sum(1)) + _H(j.sum(0)) - _H(j.ravel()))
    return total


def responsiveness(E: pd.DataFrame, var: str, n_shuffles: int, seed: int = 0):
    """A_u = I(L;u|X)/H(L), minus a null that reassigns u across responses.

    The permutation is done *within corpus*, because the user-state marginals
    differ by corpus and we do not want corpus identity leaking in as signal.
    """
    d = E.dropna(subset=[var]).copy()
    labels = sorted(d.label.unique())
    lab = d.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    xbin = np.minimum((d.position.to_numpy() * N_BINS).astype(int), N_BINS - 1)
    ucats = sorted(d[var].astype(str).unique())
    u = d[var].astype(str).map({c: i for i, c in enumerate(ucats)}).to_numpy()
    H_L = _H(np.bincount(lab, minlength=len(labels)).astype(float))

    obs = _cond_mi(lab, u, xbin, len(labels), len(ucats), N_BINS) / H_L

    # response-level table so the null permutes a response's state as a unit
    resp = d.groupby("response_id", sort=False).agg(
        corpus=("corpus", "first"), u=(var, lambda s: str(s.iloc[0])))
    rid = d.response_id.to_numpy()
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    for s in range(n_shuffles):
        perm = resp.u.copy()
        for _, idx in resp.groupby("corpus").groups.items():
            vals = resp.loc[idx, "u"].to_numpy()
            perm.loc[idx] = rng.permutation(vals)
        umap = perm.map({c: i for i, c in enumerate(ucats)}).to_dict()
        u_s = np.array([umap[r] for r in rid])
        null[s] = _cond_mi(lab, u_s, xbin, len(labels), len(ucats), N_BINS) / H_L
    mu, sd = null.mean(), null.std()
    return dict(variable=var, A=round(obs - mu, 4),
                z=round(float((obs - mu) / sd) if sd > 0 else np.nan, 2),
                raw=round(obs, 4), null=round(float(mu), 4),
                n_levels=len(ucats), n_events=len(d),
                n_responses=int(d.response_id.nunique()))


def position_baseline(E: pd.DataFrame, n_levels: int, n_shuffles: int,
                      seed: int = 0) -> float:
    """C = I(L;X)/H(L) with X coarsened to exactly `n_levels` bins.

    A user variable with 3 levels and a position variable with 10 bins are not
    comparable — the bigger table finds more mutual information for free. This
    hands position the same number of levels, so A_u and C are measured with
    the same capacity and the same small-sample bias.
    """
    labels = sorted(E.label.unique())
    lab = E.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    xb = np.minimum((E.position.to_numpy() * n_levels).astype(int), n_levels - 1)
    rid = pd.factorize(E.response_id)[0]
    H_L = _H(np.bincount(lab, minlength=len(labels)).astype(float))

    def mi(l_):
        j = np.zeros((len(labels), n_levels))
        np.add.at(j, (l_, xb), 1.0)
        return (_H(j.sum(1)) + _H(j.sum(0)) - _H(j.ravel())) / H_L

    obs = mi(lab)
    rng = np.random.default_rng(seed)
    null = np.array([mi(sm._shuffle_within(lab, rid, rng))
                     for _ in range(n_shuffles)])
    return round(float(obs - null.mean()), 4)


# ------------------------------------------------------------------ test 2

def _profile_from(df: pd.DataFrame, labels: list) -> dict:
    l2i = {l: i for i, l in enumerate(labels)}
    lab = df.label.map(l2i).to_numpy()
    xbin = np.minimum((df.position.to_numpy() * N_BINS).astype(int), N_BINS - 1)
    rid = pd.factorize(df.response_id)[0]
    pos = np.zeros((len(labels), N_BINS))
    np.add.at(pos, (lab, xbin), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return dict(labels=labels, position=pos.tolist(), transition=tr.tolist(),
                n_bins=N_BINS)


def _shuffle_probe(probe: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute the probe's own labels across its own positions.

    Composition and positions are both preserved, so anything the shuffled
    probe still explains is composition, not arrangement — the same logic as
    SCRIPT's null, applied to a single reply.
    """
    out = probe.copy()
    out["label"] = rng.permutation(probe.label.to_numpy())
    return out


def head_to_head(E: pd.DataFrame, draws: int, seed: int = 0, n_shuf: int = 30):
    """Same-prompt reference vs same-model reference, four replies each.

    Reports two margins per reply:

      raw          ll(same-model) - ll(same-prompt) on the real reply. Mixes
                   *which* behaviours appear (composition) with *where* they
                   sit (arrangement).
      arrangement  the same margin minus its average over within-reply label
                   shuffles. Composition is identical in a shuffled reply, so
                   it cancels and only the arrangement contribution is left.
    """
    labels = sorted(E.label.unique())
    by_resp = {r: g for r, g in E.groupby("response_id", sort=False)}
    meta = (E.groupby("response_id", sort=False)
            .agg(item=("item", "first"), model=("model", "first")))
    by_item = meta.groupby("item").groups          # prompt -> its 5 replies
    by_model = meta.groupby("model").groups
    rng = np.random.default_rng(seed)

    rows = []
    targets = [r for r, it in meta.item.items() if len(by_item[it]) == 5]
    picks = rng.choice(targets, size=min(draws, len(targets)), replace=False)
    for target in picks:
        it, mo = meta.loc[target, "item"], meta.loc[target, "model"]
        same_prompt = [r for r in by_item[it] if r != target]
        pool = [r for r in by_model[mo] if r != target]
        if len(same_prompt) != 4 or len(pool) < 4:
            continue
        same_model = list(rng.choice(pool, 4, replace=False))   # matched to 4

        probe = by_resp[target].sort_values("position")
        if len(probe) < 3:
            continue
        p_prompt = _profile_from(pd.concat([by_resp[r] for r in same_prompt]), labels)
        p_model = _profile_from(pd.concat([by_resp[r] for r in same_model]), labels)
        ll_prompt = sm.profile_loglik(probe, p_prompt)
        ll_model = sm.profile_loglik(probe, p_model)
        margin = ll_model - ll_prompt

        null = np.empty(n_shuf)
        for s in range(n_shuf):
            sh = _shuffle_probe(probe, rng)
            null[s] = (sm.profile_loglik(sh, p_model)
                       - sm.profile_loglik(sh, p_prompt))
        arr = margin - null.mean()

        rows.append(dict(response_id=target, model=mo, item=it,
                         ll_same_prompt=ll_prompt, ll_same_model=ll_model,
                         margin_raw=margin, margin_null=float(null.mean()),
                         margin_arrangement=arr,
                         model_wins_raw=margin > 0,
                         model_wins_arrangement=arr > 0,
                         n_events=len(probe)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--test", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    E = pd.read_csv(EVENTS)
    E["item"] = E.corpus + "|" + E.row.astype(str)
    E["response_id"] = E.item + "|" + E.model

    # attach the clinician's ratings of the user's turn
    meta = []
    for corpus in CORPORA:
        df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        cols = [c for c in USER_VARS if c in df.columns]
        m = df[cols].copy()
        m["item"] = corpus + "|" + m.index.astype(str)
        meta.append(m)
    M = pd.concat(meta, ignore_index=True)
    E = E.merge(M, on="item", how="left")
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
    print(f"{len(E)} events, {E.response_id.nunique()} responses, "
          f"{E.item.nunique()} prompts\n")

    if args.test in ("1", "both"):
        print("=" * 74)
        print("TEST 1 — responsiveness to the user's clinician-rated state")
        print("=" * 74)
        print("A_u = I(L;u|X)/H(L) in excess of a user-state permutation null.")
        print("Compare against SCRIPT's own C = I(L;X)/H(L) ~ 0.055.\n")
        out = []
        for scope, sub in [("ALL", E)] + [(m, E[E.model == m]) for m in MODELS]:
            for var in USER_VARS:
                if var not in sub.columns:
                    continue
                r = responsiveness(sub, var, args.shuffles)
                # Matched-capacity control: give POSITION exactly as many
                # levels as this user variable has, so the two predictors have
                # identical table sizes and identical bias. Anything left is a
                # like-for-like comparison of "where it sits" against "what the
                # user needed".
                r["C_matched"] = position_baseline(sub, r["n_levels"],
                                                   args.shuffles)
                r["ratio_C_over_A"] = (round(r["C_matched"] / r["A"], 1)
                                       if r["A"] > 0 else np.nan)
                r["scope"] = scope
                out.append(r)
                if scope == "ALL":
                    print(f"  {var:22s} A={r['A']:+.4f} (z={r['z']:5.1f})   "
                          f"C@{r['n_levels']}lv={r['C_matched']:+.4f}   "
                          f"C/A={r['ratio_C_over_A']:>5}   {USER_VARS[var]}")
        T1 = pd.DataFrame(out)
        T1.to_csv(OUTDIR / "user_responsiveness.csv", index=False)
        print(f"\n  pooled |A| max = {T1[T1.scope=='ALL'].A.abs().max():.4f}")
        print(f"  wrote {OUTDIR / 'user_responsiveness.csv'}\n")

    if args.test in ("2", "both"):
        print("=" * 74)
        print("TEST 2 — matched head-to-head, 4 reference replies either side")
        print("=" * 74)
        H = head_to_head(E, args.draws)
        H.to_csv(OUTDIR / "prompt_vs_model.csv", index=False)

        def report(col, win_col, title):
            m, s, n = H[col].mean(), H[col].std(ddof=1), len(H)
            t = m / (s / np.sqrt(n))
            print(f"\n  {title}")
            print(f"    same-model reference wins : {H[win_col].mean():.1%}"
                  f"   (chance 50%)")
            print(f"    mean margin               : {m:+.4f} nats/event "
                  f"(paired t = {t:+.1f}, n = {n})")

        print(f"  {len(H)} held-out replies, 4 reference replies either side")
        report("margin_raw", "model_wins_raw",
               "RAW  (composition + arrangement)")
        report("margin_arrangement", "model_wins_arrangement",
               "ARRANGEMENT ONLY  (within-reply shuffle removes composition)")
        print("\n  by model, arrangement only:")
        for mo, g in H.groupby("model"):
            print(f"    {mo:8s} model-wins {g.model_wins_arrangement.mean():5.1%}"
                  f"   margin {g.margin_arrangement.mean():+.4f}   (n={len(g)})")
        print(f"\n  wrote {OUTDIR / 'prompt_vs_model.csv'}")


if __name__ == "__main__":
    main()
