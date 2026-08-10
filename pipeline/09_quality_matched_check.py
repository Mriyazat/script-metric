"""Are the SCRIPT-vs-expert-quality gaps sample-size artefacts?

Both quality claims in the ablation table compare a large stratum against a
small one: the accurate-empathy (EMP=2) and the unflagged (harmful=0) strata
carry several times fewer events than their counterparts. SCRIPT is
downward-biased at small n (see operating guidance), so the smaller stratum is
expected to read lower whatever its behaviour. Here we subsample both strata to
a common response count and re-score, which removes the bias from the contrast.

Writes tables/quality_matched.csv. Needs the downloaded corpora for the
expert ratings, so run 00_get_data.py first.

Each axis takes a few minutes; run them separately and merge if that suits
your machine better:

    python 09_quality_matched_check.py                  # both axes
    python 09_quality_matched_check.py "empathy accuracy"
    python 09_quality_matched_check.py "harmful flag"
"""
import sys

import numpy as np
import pandas as pd

from paths import EVENTS, DATA_DIR, TAB, MODELS, MODNUM, CORPORA

import script_metric as sm

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model

meta = {}
for corpus in CORPORA:
    df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv")
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    for i, r in df.iterrows():
        for m, n in MODNUM.items():
            meta[f"{corpus}|{i}|{m}"] = (r.get(f"Response {n}_EMP_score"),
                                         r.get(f"Response {n}_yn_harmful"))
E["emp"] = E.response_id.map(lambda k: meta.get(k, (np.nan, np.nan))[0])
E["harm"] = E.response_id.map(lambda k: meta.get(k, (np.nan, np.nan))[1])
print("rating coverage:", E.emp.notna().mean().round(3))

DRAWS = 20


def score(d, shuffles=200, seed=0):
    d = d[["response_id", "label", "position"]].sort_values(
        ["response_id", "position"]).reset_index(drop=True)
    return sm.compute(d, 10, shuffles, seed)[0]


def subframe(groups, picks):
    """Assemble n responses into a frame, keeping duplicates distinct so that
    a bootstrap draw is not silently merged into one giant response."""
    parts = []
    for j, r in enumerate(picks):
        g = groups[r].copy()
        g["response_id"] = f"{r}#{j}"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def matched(d, n):
    """SCRIPT at a matched size of n responses, with a sampling spread.

    The larger stratum is subsampled without replacement (this is what removes
    the small-n bias from the contrast); the stratum already at size n is
    scored whole and its spread comes from a bootstrap, so both sides of the
    comparison carry comparable error bars.
    """
    groups = {r: g for r, g in d.groupby("response_id")}
    rids = np.array(list(groups))
    rng = np.random.default_rng(0)
    boot = len(rids) == n
    vals = [score(subframe(groups, rng.choice(rids, n, replace=boot)), 100, b)
            ["SCRIPT"] for b in range(DRAWS)]
    point = score(d)["SCRIPT"] if boot else float(np.mean(vals))
    return point, float(np.std(vals))


# each axis: (name, "good" stratum mask, "bad" stratum mask). The published
# claim in every case is that the good stratum is the less scripted one.
AXES = [("empathy accuracy", lambda d: d.emp == 2, lambda d: d.emp == 0,
         "accurate", "inaccurate"),
        ("harmful flag", lambda d: d.harm == 0, lambda d: d.harm == 1,
         "unflagged", "flagged")]

WANTED = sys.argv[1] if len(sys.argv) > 1 else None
OUT_CSV = TAB / "quality_matched.csv"

rows = []
if WANTED and OUT_CSV.exists():          # keep whatever the other run produced
    rows = pd.read_csv(OUT_CSV).query("axis != @WANTED").to_dict("records")

for axis, good_m, bad_m, gname, bname in AXES:
    if WANTED and axis != WANTED:
        continue
    print(f"\n=== {axis}: is '{gname} is less scripted' real? ===")
    print(f"{'model':7s} {'n resp':>11s} | {'as published':>24s} "
          f"| {'at matched n':>26s} | verdict")
    for m in MODELS:
        d = E[E.model == m]
        good, bad = d[good_m(d)], d[bad_m(d)]
        n_g, n_b = good.response_id.nunique(), bad.response_id.nunique()
        pub_g, pub_b = score(good)["SCRIPT"], score(bad)["SCRIPT"]

        n = min(n_g, n_b)
        mg, sg = matched(good, n)
        mb, sb = matched(bad, n)
        d_pub, d_match = pub_g - pub_b, mg - mb
        pooled = float(np.hypot(sg, sb))
        z = d_match / pooled if pooled > 0 else float("nan")
        verdict = ("survives" if z < -1.5 else
                   "reversed" if z > 1.5 else "not resolved")
        print(f"{m:7s} {n_g:>4d}/{n_b:<6d} | {gname[:5]} {pub_g:.4f} "
              f"{bname[:5]} {pub_b:.4f} d={d_pub:+.4f} | {mg:.4f} vs {mb:.4f} "
              f"d={d_match:+.4f} (z={z:+.1f}) | {verdict}")
        rows.append(dict(axis=axis, model=m,
                         n_resp_good=n_g, n_resp_bad=n_b,
                         n_events_good=len(good), n_events_bad=len(bad),
                         SCRIPT_good=round(pub_g, 4), SCRIPT_bad=round(pub_b, 4),
                         delta_published=round(d_pub, 4),
                         matched_n_responses=n,
                         SCRIPT_good_matched=round(mg, 4), sd_good=round(sg, 4),
                         SCRIPT_bad_matched=round(mb, 4), sd_bad=round(sb, 4),
                         delta_matched=round(d_match, 4), z_matched=round(z, 2),
                         verdict=verdict))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print("\nwrote", OUT_CSV)
print("\nsign of the gap, published vs matched:")
print(out.pivot_table(index="model", columns="axis",
                      values=["delta_published", "delta_matched"]).round(4))
