#!/usr/bin/env python3
"""How well the profile names an unknown generator, and what breaks it.

The identification game: hide the model name on k annotated replies, score
them against each enrolled reference profile by mean per-event log-likelihood,
and take the argmax. Chance is 1/5 = 0.20.

Five generalisation settings, from easy to adversarial:

  within corpus       enrol and probe inside one corpus (disjoint replies)
  held-out corpus     enrol on three corpora, probe the fourth
  held-out annotator  enrol on five annotators, probe the sixth
  both new            a corpus AND an annotator the reference never saw
  permutation null    labels globally permuted through the same pipeline;
                      this must land at chance, or the protocol leaks

    python 14_identification_curves.py [--draws 200]

Writes tables/identification_curves.csv (long form: setting x k x accuracy)
and derived/identification_curves.csv for the identity figure.
"""
import argparse

import numpy as np
import pandas as pd

from paths import DERIVED, EVENTS, MODELS, TAB

import script_metric as sm

K_VALUES = [1, 5, 10, 20]
N_BINS = 10
CHANCE = 1.0 / len(MODELS)


def build_profile(df: pd.DataFrame) -> dict:
    """Position + transition count tables — the profile, without the score."""
    labels = sorted(df.label.astype(str).unique())
    l2i = {l: i for i, l in enumerate(labels)}
    lab = df.label.astype(str).map(l2i).to_numpy()
    xbin = np.minimum((df.position.to_numpy() * N_BINS).astype(int), N_BINS - 1)
    rid = pd.factorize(df.response_id)[0]
    pos = np.zeros((len(labels), N_BINS))
    np.add.at(pos, (lab, xbin), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return dict(labels=labels, position=pos.tolist(), transition=tr.tolist(),
                n_bins=N_BINS)


def _per_response_loglik(pool: pd.DataFrame, ref: dict, alpha: float = 0.5):
    """Total log-likelihood and event count of every response under `ref`.

    ``sm.profile_loglik`` averages log P(label | bin) + log P(label | prev)
    over the probe's events, and the previous-label term never crosses a
    response boundary. So a probe's score decomposes exactly into per-response
    sums, which lets a draw be scored in O(k) instead of rescanning its events:

        score(probe) = sum_r total[r] / sum_r n[r]

    Same arithmetic as ``sm.profile_loglik``, just precomputed once.
    """
    labels = ref["labels"]
    l2i = {l: i for i, l in enumerate(labels)}
    nb = ref["n_bins"]
    P = np.array(ref["position"]) + alpha
    T = np.array(ref["transition"]) + alpha
    P = P / P.sum(0, keepdims=True)      # P(label | bin)
    T = T / T.sum(1, keepdims=True)      # P(current | previous)
    logP, logT = np.log(P), np.log(T)
    backoff = np.log(1.0 / len(labels))

    d = pool.sort_values(["response_id", "position"])
    rid = d.response_id.to_numpy()
    lab = np.array([l2i.get(str(x), -1) for x in d.label.to_numpy()])
    bins = np.minimum((d.position.to_numpy() * nb).astype(int), nb - 1)

    totals, counts = {}, {}
    prev = -1
    prev_rid = None
    for r, li, b in zip(rid, lab, bins):
        if r != prev_rid:
            prev, prev_rid = -1, r
        val = logP[li, b] if li >= 0 else backoff
        if prev >= 0 and li >= 0:
            val += logT[prev, li]
        totals[r] = totals.get(r, 0.0) + val
        counts[r] = counts.get(r, 0) + 1
        prev = li
    return totals, counts


def play(E: pd.DataFrame, enrol_mask, probe_mask, k: int, draws: int,
         seed: int) -> tuple[int, int]:
    """Enrol on one slice, probe from another. -> (correct, played)."""
    refs = {}
    for m in MODELS:
        d = E[enrol_mask & (E.model == m)]
        if len(d) < 200:
            return 0, 0
        refs[m] = build_profile(d)
    rng = np.random.default_rng(seed)
    correct = played = 0
    for true_m in MODELS:
        pool = E[probe_mask & (E.model == true_m)]
        rids = pool.response_id.unique()
        if len(rids) < k:
            continue
        order = {r: i for i, r in enumerate(rids)}
        tot = np.zeros((len(MODELS), len(rids)))
        cnt = np.zeros(len(rids))
        for mi, m in enumerate(MODELS):
            totals, counts = _per_response_loglik(pool, refs[m])
            for r, v in totals.items():
                tot[mi, order[r]] = v
            if mi == 0:
                for r, c in counts.items():
                    cnt[order[r]] = c
        for _ in range(draws):
            idx = rng.choice(len(rids), k, replace=False)
            n = cnt[idx].sum()
            if n == 0:
                continue
            scores = tot[:, idx].sum(1) / n
            correct += MODELS[int(np.argmax(scores))] == true_m
            played += 1
    return correct, played


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200,
                    help="probe draws per (model, split); 200 in the paper")
    args = ap.parse_args()

    E = pd.read_csv(EVENTS)
    E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
    corpora = sorted(E.corpus.unique())
    # annotators heavy enough to enrol a reference from
    reviewers = [r for r, n in E.reviewer.value_counts().items()
                 if str(r).startswith("R") and n >= 1000]
    print(f"{len(E)} events | corpora {corpora} | annotators {reviewers}\n")

    # a globally label-permuted copy: identical shapes, no real identity left
    rng = np.random.default_rng(12345)
    Eperm = E.copy()
    Eperm["label"] = rng.permutation(E.label.to_numpy())

    rows = []
    for k in K_VALUES:
        # ---- within corpus: enrol on odd-indexed replies, probe the even ones
        c_ok = c_n = 0
        for corpus in corpora:
            sub = E[E.corpus == corpus]
            rids = sorted(sub.response_id.unique())
            half = set(rids[::2])
            enrol = E.corpus.eq(corpus) & E.response_id.isin(half)
            probe = E.corpus.eq(corpus) & ~E.response_id.isin(half)
            a, b = play(E, enrol, probe, k, args.draws, seed=1000 + k)
            c_ok += a
            c_n += b
        rows.append(dict(setting="within corpus", k=k, correct=c_ok, games=c_n))

        # ---- held-out corpus
        c_ok = c_n = 0
        for held in corpora:
            a, b = play(E, E.corpus.ne(held), E.corpus.eq(held), k,
                        args.draws, seed=2000 + k)
            c_ok += a
            c_n += b
        rows.append(dict(setting="held-out corpus", k=k, correct=c_ok, games=c_n))

        # ---- held-out annotator
        c_ok = c_n = 0
        for held in reviewers:
            a, b = play(E, E.reviewer.ne(held), E.reviewer.eq(held), k,
                        args.draws, seed=3000 + k)
            c_ok += a
            c_n += b
        rows.append(dict(setting="held-out annotator", k=k, correct=c_ok, games=c_n))

        # ---- both new: unseen corpus AND unseen annotator
        c_ok = c_n = 0
        for held_c in corpora:
            for held_r in reviewers:
                enrol = E.corpus.ne(held_c) & E.reviewer.ne(held_r)
                probe = E.corpus.eq(held_c) & E.reviewer.eq(held_r)
                a, b = play(E, enrol, probe, k, args.draws, seed=4000 + k)
                c_ok += a
                c_n += b
        rows.append(dict(setting="both new", k=k, correct=c_ok, games=c_n))

        # ---- permutation null, run through the held-out-corpus protocol
        c_ok = c_n = 0
        for held in corpora:
            a, b = play(Eperm, Eperm.corpus.ne(held), Eperm.corpus.eq(held), k,
                        args.draws, seed=5000 + k)
            c_ok += a
            c_n += b
        rows.append(dict(setting="permutation null", k=k, correct=c_ok, games=c_n))
        print(f"k={k} done", flush=True)

    T = pd.DataFrame(rows)
    T["accuracy"] = (T.correct / T.games.clip(lower=1)).round(4)
    T["chance"] = CHANCE
    T.to_csv(TAB / "identification_curves.csv", index=False)
    T.to_csv(DERIVED / "identification_curves.csv", index=False)

    print("\n== rank-1 identification accuracy (chance = 0.20) ==")
    print(T.pivot(index="setting", columns="k", values="accuracy").to_string())

    null_max = T[T.setting == "permutation null"].accuracy.max()
    if null_max > 0.30:
        print(f"\nWARNING: permutation null reaches {null_max:.2f}, well above "
              f"chance — the protocol is leaking identity.")
    print(f"\nwrote {TAB / 'identification_curves.csv'}")


if __name__ == "__main__":
    main()
