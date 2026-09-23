#!/usr/bin/env python3
"""Per-reply conformity to the system's own blueprint, and whether it predicts rated quality.

conformity(reply) = mean per-event log-likelihood of the reply's (label, position) events under the
system's profile built from all OTHER replies (leave-one-out; Eq. identify of the paper), minus the
mean of the same quantity over K within-reply label shuffles of the reply.  The shuffle keeps the
reply's label multiset, positions and slot structure, so the excess isolates ARRANGEMENT, exactly
as the SCRIPT null does at system level.

Two testbeds, two outcome sets:
  Testbed 1  clinician flags and attribute scores of each model reply (Cognitive Atrophy Benchmark)
  Testbed 2  the expert-protocol empathy rating of each supporter turn (Zhan et al., 22 systems)

For each, two readings:
  raw      Spearman between conformity excess and the outcome within each system, and pooled on
           within-system percentile ranks
  matched  pairs of replies from the same system (and corpus) with the SAME label multiset and
           different arrangement; the higher-conformity reply minus the lower on each outcome,
           tested by sign-flip at the level of matched groups (a reply may enter several pairs)

Writes tables/conformity_tb1_raw.csv, conformity_tb1_pairs.csv, conformity_tb2_raw.csv,
conformity_tb2_pairs.csv, conformity_summary.json.
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.common import benchmark as bm
from pipeline.common.paths import DERIVED, EVENTS, GROUPS4, MINT_DIR, MODELS, TAB

N_BINS, K, ALPHA, MIN_EVENTS, SEED, N_PERM = 10, 50, 0.5, 3, 0, 4000
OUTCOMES_TB1 = ["yn_harmful", "yn_assumes", "yn_decisive", "yn_introduces",
                "EMP", "QOC", "TN", "AUR", "TD", "FIXbin"]
G13 = {**{c: "EA" for c in ["VAC", "NAC", "ASAC", "SAC"]}, **{c: "EI" for c in ["VIN", "NIN", "ASIN", "SIN"]}}


def profile_counts(ev, labels, n_bins=N_BINS):
    """Unsmoothed position and transition count tables of a set of events (slot convention)."""
    l2i = {l: i for i, l in enumerate(labels)}
    ev = ev.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
    lab = ev.label.astype(str).map(l2i).to_numpy(); pos = ev.position.to_numpy()
    xb = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    rid = pd.factorize(ev.response_id)[0]
    trans = (rid[1:] == rid[:-1]) & (pos[1:] > pos[:-1])
    P = np.zeros((len(labels), n_bins)); np.add.at(P, (lab, xb), 1.0)
    T = np.zeros((len(labels), len(labels))); np.add.at(T, (lab[:-1][trans], lab[1:][trans]), 1.0)
    return P, T


def per_reply_conformity(ev, n_bins=N_BINS, k=K, alpha=ALPHA, min_events=MIN_EVENTS, seed=SEED, reference=None):
    """ev: response_id, label, position for ONE system. One row per reply.
    reference: optional (labels, P, T) count tables of a shared profile (e.g. all systems pooled);
    the scored reply's own counts are removed from it (leave-one-out) exactly as for the own profile."""
    ev = ev.sort_values(["response_id", "position"], kind="stable").reset_index(drop=True)
    if reference is None:
        labels = sorted(ev.label.astype(str).unique())
        P, T = profile_counts(ev, labels, n_bins)
    else:
        labels, P, T = reference
        P, T = P.copy(), T.copy()
    l2i = {l: i for i, l in enumerate(labels)}
    L = len(labels)
    lab = ev.label.astype(str).map(l2i).to_numpy()
    pos = ev.position.to_numpy()
    xb = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    rid = pd.factorize(ev.response_id)[0]
    rng = np.random.default_rng(seed)
    ids = ev.response_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    rows = []
    for s, e in zip(starts[:-1], starts[1:]):
        l, b, p = lab[s:e], xb[s:e], pos[s:e]
        n = e - s
        tm = p[1:] > p[:-1]
        Pr = np.zeros_like(P); np.add.at(Pr, (l, b), 1.0)
        Tr = np.zeros_like(T); np.add.at(Tr, (l[:-1][tm], l[1:][tm]), 1.0)
        Pl = P - Pr + alpha; Tl = T - Tr + alpha                  # leave-one-out profile
        logP = np.log(Pl / Pl.sum(0, keepdims=True)); logT = np.log(Tl / Tl.sum(1, keepdims=True))

        def ll(lv):
            return (logP[lv, b].sum() + logT[lv[:-1][tm], lv[1:][tm]].sum()) / n

        real = ll(l)
        if n < min_events:
            rows.append(dict(response_id=ids[s], n_events=n, ll=real, excess=np.nan)); continue
        null = np.array([ll(rng.permutation(l)) for _ in range(k)])
        rows.append(dict(response_id=ids[s], n_events=n, ll=real, excess=real - null.mean()))
    return pd.DataFrame(rows)


def within_rank(v, g):
    return pd.Series(np.asarray(v)).groupby(pd.Series(np.asarray(g))).rank(pct=True).to_numpy()


def pooled_rank_corr(x, y, g, rng, n_perm=1000):
    rx, ry = within_rank(x, g), within_rank(y, g)
    obs = np.corrcoef(rx, ry)[0, 1]
    idx = [np.flatnonzero(np.asarray(g) == v) for v in np.unique(g)]
    cnt = 0
    for _ in range(n_perm):
        xp = rx.copy()
        for ix in idx:
            xp[ix] = rng.permutation(rx[ix])
        cnt += abs(np.corrcoef(xp, ry)[0, 1]) >= abs(obs)
    return obs, (cnt + 1) / (n_perm + 1)


def matched_pairs(D, keycols, outcomes):
    rows = []
    for key, g in D.groupby(keycols):
        if len(g) < 2:
            continue
        a = g.sort_values("excess").to_dict("records")
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if a[j]["excess"] - a[i]["excess"] <= 0:
                    continue
                rows.append(dict(group="|".join(map(str, key)), d_excess=a[j]["excess"] - a[i]["excess"],
                                 id_lo=a[i]["response_id"], id_hi=a[j]["response_id"],
                                 **{oc: a[j][oc] - a[i][oc] for oc in outcomes}))
    return pd.DataFrame(rows)


def matched_vs_rest(D, P, sys_col):
    ids = set(P.id_lo) | set(P.id_hi)
    m = D[sys_col].astype(str) + "|" + D.response_id.astype(str)
    key_ids = set(P.group.str.split("|").str[0] + "|" + P.id_lo.astype(str)) | set(P.group.str.split("|").str[0] + "|" + P.id_hi.astype(str))
    inm = m.isin(key_ids)
    return dict(n_matched=int(inm.sum()), n_unmatched=int((~inm).sum()),
                events_matched=round(float(D[inm].n_events.mean()), 1), events_unmatched=round(float(D[~inm].n_events.mean()), 1),
                excess_matched=round(float(D[inm].excess.mean()), 3), excess_unmatched=round(float(D[~inm].excess.mean()), 3),
                sd_excess_matched=round(float(D[inm].excess.std()), 3), sd_excess_unmatched=round(float(D[~inm].excess.std()), 3))


def group_signflip(P, outcomes, rng, n_perm=N_PERM, sesoi=None):
    """Group-level sign-flip test, cluster-bootstrap 95% CI over matched groups, and the largest
    effect the CI excludes (the equivalence bound: |effect| <= bound at the 95% level)."""
    out = []
    for oc in outcomes:
        g = P.groupby("group")[oc].mean().dropna().to_numpy()
        obs = g.mean()
        signs = rng.choice([-1, 1], size=(n_perm, len(g)))
        p = ((np.abs((signs * g).mean(1)) >= abs(obs)).sum() + 1) / (n_perm + 1)
        boot = np.array([g[rng.integers(0, len(g), len(g))].mean() for _ in range(n_perm)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nz = P[oc][P[oc] != 0]
        row = dict(outcome=oc, groups=len(g), pairs=int(P[oc].notna().sum()), pairs_nontied=len(nz),
                   mean_diff=round(obs, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), bound=round(max(abs(lo), abs(hi)), 4),
                   frac_higher_conformity_higher_value=round((nz > 0).mean(), 3) if len(nz) else np.nan,
                   p_group_signflip=round(p, 4))
        if sesoi is not None:   # TOST: both one-sided tests at alpha=0.05 via the 90% bootstrap interval
            l90, h90 = np.percentile(boot, [5, 95])
            row["tost_equivalent_at"] = sesoi; row["tost_pass"] = bool(l90 > -sesoi and h90 < sesoi)
        out.append(row)
    return pd.DataFrame(out)


def range_diagnostics(D, P, sys_col):
    """Restriction of range: how much conformity varies inside matched pairs, in units of the
    corpus-wide (within-system) SD of conformity, and how matched replies differ from the rest."""
    sd = D.groupby(sys_col).excess.transform("std")
    D = D.assign(sd=sd)
    sd_pooled = float(D.groupby(sys_col).excess.std().median())
    matched_ids = set()
    for grp in P.group.unique():
        pass
    return dict(within_system_sd_conformity=round(sd_pooled, 3),
                mean_pair_diff=round(float(P.d_excess.mean()), 3),
                median_pair_diff=round(float(P.d_excess.median()), 3),
                mean_pair_diff_in_sd=round(float(P.d_excess.mean() / sd_pooled), 2),
                q75_pair_diff_in_sd=round(float(P.d_excess.quantile(0.75) / sd_pooled), 2),
                q90_pair_diff_in_sd=round(float(P.d_excess.quantile(0.90) / sd_pooled), 2))


SESOI_TB1, SESOI_TB2 = 0.10, 0.10     # smallest effect of interest: 0.1 on a 0-2 score / 10 flag points; 0.1 on the 1-5 rating


def testbed1(rng):
    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str)
    labels_all = sorted(E.label.astype(str).unique())
    Ppool, Tpool = profile_counts(E.assign(response_id=E.response_id + "|" + E.model)[["response_id", "label", "position"]], labels_all)
    F = bm.flags_long().pivot_table(index=["corpus", "row", "model"], columns="flag", values="value").reset_index()
    S = bm.scores_long().pivot_table(index=["corpus", "row", "model"], columns="attribute", values="value").reset_index()
    E["g5"] = E.label.map(GROUPS4).fillna("other"); E["g13"] = E.label.map(lambda l: G13.get(l, l))
    keys = E.groupby(["corpus", "row", "model"]).agg(
        key5=("g5", lambda s: str(sorted(Counter(s).items()))),
        key13=("g13", lambda s: str(sorted(Counter(s).items())))).reset_index()
    summary = {}
    for mode in ("own", "pooled"):
        ref = None if mode == "own" else (labels_all, Ppool, Tpool)
        C = pd.concat([per_reply_conformity(E[E.model == m][["response_id", "label", "position"]], reference=ref).assign(model=m)
                       for m in MODELS], ignore_index=True)
        C[["corpus", "row"]] = C.response_id.str.split("|", expand=True); C["row"] = C.row.astype(int)
        D = C.merge(F, on=["corpus", "row", "model"]).merge(S, on=["corpus", "row", "model"]).merge(keys, on=["corpus", "row", "model"])
        D["FIXbin"] = (D.FIX > 0).astype(float)
        D = D.dropna(subset=["excess"])
        pos_share = float((D.excess > 0).mean())
        print(f"\n===== Testbed 1, {mode} profile: {len(D)} replies with >= {MIN_EVENTS} events; share with positive excess "
              f"{pos_share:.3f}; Spearman(excess, n_events) {spearmanr(D.excess, D.n_events)[0]:.2f}")
        raw = []
        for oc in OUTCOMES_TB1:
            for m, d in D.groupby("model"):
                d = d.dropna(subset=[oc]); rho, p = spearmanr(d.excess, d[oc])
                raw.append(dict(profile=mode, outcome=oc, model=m, n=len(d), rho=round(rho, 3), p=round(p, 4)))
            d = D.dropna(subset=[oc])
            rho, p = pooled_rank_corr(d.excess.to_numpy(), d[oc].to_numpy(), d.model.to_numpy(), rng)
            raw.append(dict(profile=mode, outcome=oc, model="POOLED", n=len(d), rho=round(rho, 3), p=round(p, 4)))
        R = pd.DataFrame(raw)
        P5 = matched_pairs(D, ["model", "corpus", "key5"], OUTCOMES_TB1)
        P13 = matched_pairs(D, ["model", "corpus", "key13"], OUTCOMES_TB1)
        T5 = group_signflip(P5, OUTCOMES_TB1, rng, sesoi=SESOI_TB1).assign(matching="5 groups", profile=mode)
        T13 = group_signflip(P13, OUTCOMES_TB1, rng, sesoi=SESOI_TB1).assign(matching="13 codes", profile=mode)
        Tm = pd.concat([T5, T13])
        diag5 = range_diagnostics(D, P5, "model"); diag5.update(matched_vs_rest(D, P5, "model"))
        diag13 = range_diagnostics(D, P13, "model"); diag13.update(matched_vs_rest(D, P13, "model"))
        print("raw pooled:", {r.outcome: (r.rho, r.p) for r in R[R.model == "POOLED"].itertuples()})
        print(f"matched 5 groups: {len(P5)} pairs / {P5.group.nunique()} groups"); print(T5.drop(columns=["profile", "matching"]).to_string(index=False))
        print(f"matched 13 codes: {len(P13)} pairs / {P13.group.nunique()} groups"); print(T13.drop(columns=["profile", "matching"]).to_string(index=False))
        print("range diagnostics 5 groups:", diag5); print("range diagnostics 13 codes:", diag13)
        R.to_csv(TAB / f"conformity_tb1_raw_{mode}.csv", index=False); Tm.to_csv(TAB / f"conformity_tb1_pairs_{mode}.csv", index=False)
        D.to_csv(DERIVED / f"conformity_tb1_replies_{mode}.csv", index=False)
        summary[mode] = dict(n_replies=int(len(D)), positive_share=round(pos_share, 3),
                             pairs5=int(len(P5)), groups5=int(P5.group.nunique()), pairs13=int(len(P13)), groups13=int(P13.group.nunique()),
                             max_abs_diff5=float(T5.mean_diff.abs().max()), max_bound5=float(T5.bound.max()), min_p5=float(T5.p_group_signflip.min()),
                             max_abs_diff13=float(T13.mean_diff.abs().max()), max_bound13=float(T13.bound.max()), min_p13=float(T13.p_group_signflip.min()),
                             tost_pass5=int(T5.tost_pass.sum()), tost_pass13=int(T13.tost_pass.sum()),
                             range5=diag5, range13=diag13,
                             pooled_raw={r.outcome: (r.rho, r.p) for r in R[R.model == "POOLED"].itertuples()})
    return summary


def testbed2(rng):
    E = pd.read_csv(DERIVED / "mint_surface_events.csv")
    E = E[E.system != "gold"]
    MINT = MINT_DIR / "evaluation"
    labels_all = sorted(E.label.astype(str).unique())
    Ppool, Tpool = profile_counts(E.assign(response_id=E.system + "|" + E.response_id)[["response_id", "label", "position"]], labels_all)
    keys = E.groupby(["system", "response_id"]).label.apply(lambda s: str(sorted(Counter(s).items()))).rename("key").reset_index()
    ratings = {}
    for sysname in E.system.unique():
        rf = MINT / "eval_outputs" / sysname / "gpt-oss-120b_ratings.csv"
        if not rf.exists():
            continue
        conv = json.load(open(MINT / "outputs" / sysname / "conversations.json"))
        tidx = {f"{e['conversation_id']}|{k}": (str(e["conversation_id"]), len(e["conversation_history"])) for k, e in enumerate(conv)}
        rat = pd.read_csv(rf, dtype={"conversation_id": str}).set_index(["conversation_id", "turn_index"]).aggregated_empathy
        ratings[sysname] = (tidx, rat)
    summary = {}
    for mode in ("own", "pooled"):
        ref = None if mode == "own" else (labels_all, Ppool, Tpool)
        parts = []
        for sysname, ev in E.groupby("system"):
            if sysname not in ratings:
                continue
            tidx, rat = ratings[sysname]
            c = per_reply_conformity(ev[["response_id", "label", "position"]], reference=ref)
            c["empathy"] = [rat.get(tidx[r], np.nan) for r in c.response_id]; c["system"] = sysname
            parts.append(c)
        D = pd.concat(parts, ignore_index=True).merge(keys, on=["system", "response_id"]).dropna(subset=["excess", "empathy"])
        D["family"] = np.where(D.system.str.startswith("baseline") & ~D.system.str.contains("psychocounsel|r1zerodiv"), "prompting", "RL")
        pos_share = float((D.excess > 0).mean())
        raw = []
        for sname, d in D.groupby("system"):
            rho, p = spearmanr(d.excess, d.empathy)
            raw.append(dict(profile=mode, system=sname, family=d.family.iloc[0], n_turns=len(d), rho=round(rho, 3), p=round(p, 4),
                            rho_nevents_empathy=round(spearmanr(d.n_events, d.empathy)[0], 3)))
        for fam, d in list(D.groupby("family")) + [("ALL", D)]:
            rho, p = pooled_rank_corr(d.excess.to_numpy(), d.empathy.to_numpy(), d.system.to_numpy(), rng)
            raw.append(dict(profile=mode, system=f"POOLED {fam}", family=fam, n_turns=len(d), rho=round(rho, 3), p=round(p, 4), rho_nevents_empathy=np.nan))
        R = pd.DataFrame(raw)
        out, diag = [], {}
        for fam, d in list(D.groupby("family")) + [("ALL", D)]:
            P = matched_pairs(d, ["system", "key"], ["empathy"])
            out.append(group_signflip(P, ["empathy"], rng, sesoi=SESOI_TB2).assign(family=fam, profile=mode))
            if fam == "ALL":
                diag = range_diagnostics(d, P, "system"); diag.update(matched_vs_rest(d, P, "system"))
                # effect restricted to the pairs that differ most in conformity (top half of d_excess)
                Ph = P[P.d_excess >= P.d_excess.median()]
                out.append(group_signflip(Ph, ["empathy"], rng, sesoi=SESOI_TB2).assign(family="ALL, top-half pair gap", profile=mode))
        Tm = pd.concat(out)
        Rs = R[~R.system.str.startswith("POOLED")]
        print(f"\n===== Testbed 2, {mode} profile: {len(D)} turns; positive excess {pos_share:.3f}")
        print("raw:", R[R.system.str.startswith("POOLED")][["system", "rho", "p"]].to_string(index=False))
        print(f"systems with p<0.05: positive {int(((Rs.p < 0.05) & (Rs.rho > 0)).sum())}, negative {int(((Rs.p < 0.05) & (Rs.rho < 0)).sum())}; "
              f"median rho prompting {Rs[Rs.family == 'prompting'].rho.median():.2f}, RL {Rs[Rs.family == 'RL'].rho.median():.2f}")
        print(Tm.drop(columns=["profile"]).to_string(index=False)); print("range diagnostics:", diag)
        R.to_csv(TAB / f"conformity_tb2_raw_{mode}.csv", index=False); Tm.to_csv(TAB / f"conformity_tb2_pairs_{mode}.csv", index=False)
        D.to_csv(DERIVED / f"conformity_tb2_turns_{mode}.csv", index=False)
        summary[mode] = dict(n_turns=int(len(D)), positive_share=round(pos_share, 3),
                             n_pos_sig=int(((Rs.p < 0.05) & (Rs.rho > 0)).sum()), n_neg_sig=int(((Rs.p < 0.05) & (Rs.rho < 0)).sum()),
                             rho_median_prompting=float(Rs[Rs.family == "prompting"].rho.median()), rho_median_RL=float(Rs[Rs.family == "RL"].rho.median()),
                             rho_range=[float(Rs.rho.min()), float(Rs.rho.max())],
                             pooled={r.system: (r.rho, r.p) for r in R[R.system.str.startswith("POOLED")].itertuples()},
                             matched={r.family: dict(groups=int(r.groups), pairs=int(r.pairs), mean_diff=r.mean_diff, ci=[r.ci_lo, r.ci_hi], bound=r.bound, p=r.p_group_signflip, tost_pass=bool(r.tost_pass)) for r in Tm.itertuples()},
                             range=diag, rating_sd_within_system=round(float(D.groupby("system").empathy.std().median()), 3))
    return summary


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    summary = dict(testbed1=testbed1(rng), testbed2=testbed2(rng))
    json.dump(summary, open(TAB / "conformity_summary.json", "w"), indent=1, default=float)
    print("\nsummary written to", TAB / "conformity_summary.json")
