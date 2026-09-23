#!/usr/bin/env python3
"""A further responsiveness check on the blind LLM layer, multi-turn corpora.

1. Between-response normalisation. The person and the turn can only explain
   variation *between* responses, so we measure how much of a speaker's
   between-response variation, B = I(L; response)/H(L) in excess of a null that
   permutes labels across the speaker's responses, is accounted for by the coded
   user state and by the turn bucket. Because the therapist writes ~4.6 events per
   response against 12-18 for the models, every model is also re-scored after
   subsampling each of its responses to the therapist's per-response event-count
   distribution (10 draws).

Writes tables/listening_between.csv.
Run:  python3 -m pipeline.external.listening_extra
"""
import numpy as np
import pandas as pd

from pipeline.common.paths import DERIVED, TAB
from pipeline.external.listening import (GROUP, G5, SPEAKERS, UCOLS, _H, _mi,
                                         explained_share, prep, turn_meta)

SEED, N_PERM, N_DRAWS = 0, 300, 10


# ------------------------------------------------------ between-response share

def between_share(E, seed=SEED, n_perm=N_PERM):
    """I(L; response)/H(L) minus the null that permutes labels across all of the
    speaker's events (keeps label marginal and response sizes)."""
    hl = _H(E.label.value_counts().to_numpy())
    obs = _mi(E.label, E.rid) / hl
    rng = np.random.default_rng(seed)
    lab = E.label.to_numpy()
    null = np.array([_mi(pd.Series(rng.permutation(lab)), E.rid) / hl for _ in range(n_perm)])
    return obs - null.mean(), (obs - null.mean()) / null.std()


def budget_row(E, speaker, note):
    b, bz = between_share(E)
    pu, zu = explained_share(E, UCOLS)
    pt, zt = explained_share(E, ["tb"])
    return dict(speaker=speaker, sample=note, between=round(b, 4), between_z=round(bz, 1),
                person=pu, person_z=zu, turn=pt, turn_z=zt,
                person_over_between=round(pu / b, 3), turn_over_between=round(pt / b, 3),
                n_events=len(E), n_responses=E.rid.nunique(),
                events_per_response=round(len(E) / E.rid.nunique(), 2))


def subsample_to(E, sizes, rng):
    """Keep k events of each response, k drawn from `sizes` (the therapist's
    per-response event counts), capped at the response's own size."""
    parts = []
    for _, g in E.groupby("rid", sort=False):
        k = min(len(g), int(rng.choice(sizes)))
        parts.append(g.sample(n=k, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts).sort_values(["rid", "position"]).reset_index(drop=True)


def main():
    meta = turn_meta()
    L = prep(pd.read_csv(DERIVED / "llm_span_events.csv"), meta)
    rng = np.random.default_rng(SEED)
    rows = []
    ther = L[L.model == "Human"]
    sizes = ther.groupby("rid").size().to_numpy()
    for sp in SPEAKERS:
        E = L[L.model == sp]
        rows.append(budget_row(E, sp, "full"))
        if sp != "Human":
            draws = [budget_row(subsample_to(E, sizes, rng), sp, "density-matched") for _ in range(N_DRAWS)]
            D = pd.DataFrame(draws)
            m = D.mean(numeric_only=True).round(4).to_dict()
            m.update(speaker=sp, sample="density-matched (mean of %d draws)" % N_DRAWS,
                     n_events=int(D.n_events.mean()), n_responses=int(D.n_responses.mean()))
            rows.append(m)
        print(rows[-1], flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(TAB / "listening_between.csv", index=False)
    print(B.to_string(index=False))



if __name__ == "__main__":
    main()
