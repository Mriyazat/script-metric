#!/usr/bin/env python3
"""Does the prompt carry arrangement, once you control properly?

The first head-to-head (e1, test 2) compared two references — four replies to
the SAME PROMPT by other models, versus four replies by the SAME MODEL to other
prompts — and the same-prompt reference won. Two confounds make that
uninterpretable on its own:

  1. BASELINE. The two references are not equally good *as profiles*. The
     same-prompt set pools four different models, so it is effectively four
     samples of the shared field template and is smoother; the same-model set
     is four samples of one model's accent. Whichever is the better-estimated
     profile wins for reasons that have nothing to do with the user.

  2. LENGTH. Replies to one prompt have similar lengths, hence similar bin
     coverage to the probe. That alone can make the same-prompt profile fit
     better.

Fix for (1): add a third reference — four replies by OTHER models to OTHER
prompts — and measure both effects as a lift over that common baseline.

    lift_prompt = ll(same prompt, other models)  - ll(other prompt, other models)
    lift_model  = ll(same model,  other prompts) - ll(other prompt, other models)

Now "does the user's message carry arrangement" is lift_prompt > 0, and
"does the model carry arrangement" is lift_model > 0, each against an identical
four-reply reference built the same way. They are directly comparable.

Fix for (2): every reference set is chosen to match the same-prompt set's
per-reply event counts as closely as possible (--match length).

Composition is removed throughout by subtracting the same quantity computed on
within-reply label shuffles of the probe, exactly as in e1.

    python e1b_prompt_vs_model_controlled.py [--draws 300] [--match length|none]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from paths import EVENTS, OUT   # noqa: E402

import script_metric as sm   # noqa: E402

N_BINS = 10
N_SHUF = 25
OUTDIR = OUT / "explore"
OUTDIR.mkdir(parents=True, exist_ok=True)


def profile_from(df, labels):
    l2i = {l: i for i, l in enumerate(labels)}
    lab = df.label.map(l2i).to_numpy()
    xb = np.minimum((df.position.to_numpy() * N_BINS).astype(int), N_BINS - 1)
    rid = pd.factorize(df.response_id)[0]
    pos = np.zeros((len(labels), N_BINS))
    np.add.at(pos, (lab, xb), 1.0)
    tr = np.zeros((len(labels), len(labels)))
    same = rid[1:] == rid[:-1]
    np.add.at(tr, (lab[:-1][same], lab[1:][same]), 1.0)
    return dict(labels=labels, position=pos.tolist(), transition=tr.tolist(),
                n_bins=N_BINS)


def pick_matched(candidates, target_sizes, sizes, rng, k=4):
    """Choose k candidates whose event counts best match `target_sizes`."""
    cand = list(candidates)
    if len(cand) < k:
        return None
    chosen = []
    pool = cand[:]
    for want in sorted(target_sizes):
        best = min(pool, key=lambda r: abs(sizes[r] - want))
        chosen.append(best)
        pool.remove(best)
        if not pool:
            break
    while len(chosen) < k and pool:
        chosen.append(pool.pop(rng.integers(len(pool))))
    return chosen[:k] if len(chosen) >= k else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--match", choices=["length", "none"], default="length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    E = pd.read_csv(EVENTS)
    E["item"] = E.corpus + "|" + E.row.astype(str)
    E["response_id"] = E.item + "|" + E.model
    E = E.sort_values(["response_id", "position"]).reset_index(drop=True)
    labels = sorted(E.label.unique())

    by_resp = {r: g for r, g in E.groupby("response_id", sort=False)}
    meta = (E.groupby("response_id", sort=False)
            .agg(item=("item", "first"), model=("model", "first"),
                 corpus=("corpus", "first"), n=("label", "size")))
    sizes = meta.n.to_dict()
    by_item = {k: list(v) for k, v in meta.groupby("item").groups.items()}
    by_model = {k: list(v) for k, v in meta.groupby("model").groups.items()}
    all_ids = list(meta.index)
    rng = np.random.default_rng(args.seed)

    targets = [r for r in all_ids if len(by_item[meta.at[r, "item"]]) == 5]
    picks = rng.choice(targets, size=min(args.draws, len(targets)),
                       replace=False)

    rows = []
    for target in picks:
        it, mo = meta.at[target, "item"], meta.at[target, "model"]
        probe = by_resp[target]
        if len(probe) < 3:
            continue

        same_prompt = [r for r in by_item[it] if r != target]
        if len(same_prompt) != 4:
            continue
        target_sizes = [sizes[r] for r in same_prompt]

        # same model, different prompts
        cand_model = [r for r in by_model[mo] if meta.at[r, "item"] != it]
        # different model AND different prompt — the common baseline
        cand_other = [r for r in all_ids
                      if meta.at[r, "model"] != mo and meta.at[r, "item"] != it]
        if args.match == "length":
            same_model = pick_matched(cand_model, target_sizes, sizes, rng)
            other = pick_matched(rng.choice(cand_other, 400, replace=False),
                                 target_sizes, sizes, rng)
        else:
            same_model = list(rng.choice(cand_model, 4, replace=False))
            other = list(rng.choice(cand_other, 4, replace=False))
        if not same_model or not other:
            continue

        refs = {}
        for name, ids in [("prompt", same_prompt), ("model", same_model),
                          ("other", other)]:
            refs[name] = profile_from(pd.concat([by_resp[r] for r in ids]),
                                      labels)

        real = {k: sm.profile_loglik(probe, p) for k, p in refs.items()}
        null = {k: np.empty(N_SHUF) for k in refs}
        for s in range(N_SHUF):
            sh = probe.copy()
            sh["label"] = rng.permutation(probe.label.to_numpy())
            for k, p in refs.items():
                null[k][s] = sm.profile_loglik(sh, p)
        # arrangement-only score: real minus its own composition baseline
        arr = {k: real[k] - null[k].mean() for k in refs}

        rows.append(dict(
            response_id=target, model=mo, item=it, n_events=len(probe),
            arr_prompt=arr["prompt"], arr_model=arr["model"],
            arr_other=arr["other"],
            lift_prompt=arr["prompt"] - arr["other"],
            lift_model=arr["model"] - arr["other"],
            ref_n_prompt=sum(sizes[r] for r in same_prompt),
            ref_n_model=sum(sizes[r] for r in same_model),
            ref_n_other=sum(sizes[r] for r in other)))

    H = pd.DataFrame(rows)
    H.to_csv(OUTDIR / f"prompt_vs_model_controlled_{args.match}.csv",
             index=False)

    def stat(col):
        m, s, n = H[col].mean(), H[col].std(ddof=1), len(H)
        return m, m / (s / np.sqrt(n))

    print(f"{len(H)} held-out replies | reference size matching: {args.match}")
    print(f"mean reference events — prompt {H.ref_n_prompt.mean():.0f}, "
          f"model {H.ref_n_model.mean():.0f}, other {H.ref_n_other.mean():.0f}"
          "  (should be close if matching worked)\n")

    print("Arrangement-only lift over the common 'other model, other prompt'")
    print("baseline, in nats per event. Positive = that factor carries")
    print("arrangement information beyond a generic four-reply profile.\n")
    for col, lab in [("lift_prompt", "same PROMPT (user's message)"),
                     ("lift_model", "same MODEL")]:
        m, t = stat(col)
        share = (H[col] > 0).mean()
        print(f"  {lab:30s} lift = {m:+.4f}   t = {t:+6.1f}   "
              f"positive in {share:.0%} of replies")

    d = H.lift_prompt - H.lift_model
    m, t = d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\n  difference (prompt - model)     {m:+.4f}   t = {t:+6.1f}")
    print(f"\n  ratio of lifts  prompt/model = "
          f"{H.lift_prompt.mean() / H.lift_model.mean():.2f}")
    print(f"\nwrote {OUTDIR / f'prompt_vs_model_controlled_{args.match}.csv'}")


if __name__ == "__main__":
    main()
