#!/usr/bin/env python3
"""The interpretation ruler: SCRIPT on three unrelated annotation schemes.

The counseling score only means something against an absolute scale, so the
same estimator — same bins, same within-response permutation null, same code
path in ``script_metric.py`` — is run unchanged on three public span corpora
that have nothing to do with counseling:

  WMT24 MT error spans      2 labels  (Major / Minor)
  data-to-text error spans  6 labels  (Contradictory / NotCheckable / ...)
  RAGTruth hallucination    4 labels  (Evident/Subtle x Conflict/Baseless Info)

Nothing here is hardcoded: every value is recomputed from the downloaded
corpora. Run ``00_get_data.py`` first.

    python 11_external_anchors.py

Writes tables/anchor_wmt24.csv, anchor_d2t.csv, anchor_ragtruth.csv and the
pooled summary tables/external_anchors.csv that the ruler figure reads.
"""
import glob
import json

import numpy as np
import pandas as pd

from paths import RAGTRUTH_PATH, SPAN_ANNOTATION_DIR, TAB

import script_metric as sm

N_BINS = 10
N_SHUFFLES = 200
SEED = 0

# Kasner et al. encode the error type as an integer; these are the released
# category names, in index order.
D2T_LABELS = ["Contradictory", "NotCheckable", "Misleading", "Incoherent",
              "Repetitive", "Other"]
MT_LABELS = ["Major", "Minor"]


# --------------------------------------------------------------- span corpora

def _output_lengths(kind: str) -> dict:
    """(dataset, split, example_idx, setup_id) -> character length of the output."""
    lengths = {}
    for f in glob.glob(str(SPAN_ANNOTATION_DIR / "outputs" / kind / "*" / "*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            lengths[(d["dataset"], d["split"], int(d["example_idx"]),
                     d["setup_id"])] = len(d["output"])
    return lengths


def _load_span_events(kind: str, label_names: list[str]) -> pd.DataFrame:
    """Reduce released human annotations to (system, response_id, label, position).

    One annotated output = one response. Two annotators marking the same
    output are two responses, because the within-response null permutes
    labels inside a single annotator's reading of a single text.
    """
    lengths = _output_lengths(kind)
    ann_files = ([SPAN_ANNOTATION_DIR / "annotations" / "human" / kind / s /
                  "annotations.jsonl" for s in ("test", "dev")]
                 if kind == "d2t-eval" else
                 [SPAN_ANNOTATION_DIR / "annotations" / "human" / kind /
                  "annotations.jsonl"])
    rows = []
    for fn in ann_files:
        if not fn.exists():
            continue
        for line in open(fn):
            d = json.loads(line)
            key = (d["dataset"], d["split"], int(d["example_idx"]), d["setup_id"])
            L = lengths.get(key)
            if not L:
                continue
            group = d.get("metadata", {}).get("annotator_group", 0)
            for a in d["annotations"]:
                if a["start"] is None or a["start"] >= L:
                    continue
                rows.append(dict(system=d["setup_id"],
                                 response_key=f"{key}|{group}",
                                 label=label_names[int(a["type"])],
                                 position=a["start"] / L))
    E = pd.DataFrame(rows)
    E["response_id"] = pd.factorize(E.response_key)[0]
    return E.sort_values(["response_id", "position"]).reset_index(drop=True)


def _score_by_system(E: pd.DataFrame, scheme: str, min_events: int) -> pd.DataFrame:
    """Score every system separately, on the shared label alphabet.

    The alphabet is fixed across systems (not re-derived per system) so H(L)
    and the null are computed over the same label space everywhere.
    """
    alphabet = sorted(E.label.unique())
    rows = []
    for system, d in E.groupby("system"):
        if len(d) < min_events:
            continue
        probe = d[["response_id", "label", "position"]].copy()
        # keep unused labels in the alphabet by construction
        probe["label"] = pd.Categorical(probe.label, categories=alphabet)
        probe = probe.astype({"label": str})
        probe = probe.sort_values(["response_id", "position"]).reset_index(drop=True)
        res, _ = sm.compute(probe, n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED)
        rows.append(dict(scheme=scheme, system=system, SCRIPT=res["SCRIPT"],
                         z=res["z"], C=res["C_excess"], M=res["M_excess"],
                         n_labels=res["n_labels"], n_events=res["n_events"],
                         n_responses=res["n_responses"]))
    return (pd.DataFrame(rows).sort_values("SCRIPT", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------- RAGTruth

def _load_ragtruth() -> pd.DataFrame:
    """RAGTruth char-offset hallucination spans, all splits.

    The published anchor pools train and test: the split is a modelling
    convenience of the original hallucination-detection task and carries no
    meaning for a per-generator arrangement statistic, while the test split
    alone leaves four of the six generators under 500 events.
    """
    rows = []
    for line in open(RAGTRUTH_PATH):
        r = json.loads(line)
        if not r.get("labels"):
            continue
        L = len(r["response"])
        if L == 0:
            continue
        for lb in r["labels"]:
            rows.append(dict(system=r["model"], response_id=r["id"],
                             label=lb["label_type"],
                             position=min(lb["start"] / L, 1.0)))
    return (pd.DataFrame(rows).sort_values(["response_id", "position"])
            .reset_index(drop=True))


# -------------------------------------------------------------------- driver

def main() -> None:
    out = {}

    print("== WMT24 machine-translation error spans ==", flush=True)
    mt = _load_span_events("mt-eval", MT_LABELS)
    print(f"   {len(mt)} events, {mt.response_id.nunique()} annotated outputs, "
          f"{mt.system.nunique()} systems")
    t_mt = _score_by_system(mt, "MT error spans (WMT24)", min_events=120)
    t_mt.to_csv(TAB / "anchor_wmt24.csv", index=False)
    print(t_mt.to_string(index=False), "\n", flush=True)
    out["mt"] = t_mt

    print("== data-to-text error spans ==", flush=True)
    d2t = _load_span_events("d2t-eval", D2T_LABELS)
    print(f"   {len(d2t)} events, {d2t.response_id.nunique()} annotated outputs, "
          f"{d2t.system.nunique()} generators")
    t_d2t = _score_by_system(d2t, "Data-to-text error spans", min_events=150)
    t_d2t.to_csv(TAB / "anchor_d2t.csv", index=False)
    print(t_d2t.to_string(index=False), "\n", flush=True)
    out["d2t"] = t_d2t

    print("== RAGTruth hallucination spans ==", flush=True)
    rag = _load_ragtruth()
    print(f"   {len(rag)} events, {rag.response_id.nunique()} responses, "
          f"{rag.system.nunique()} generators, "
          f"{rag.label.nunique()} hallucination types")
    t_rag = _score_by_system(rag, "Hallucination spans (RAGTruth)", min_events=100)
    t_rag.to_csv(TAB / "anchor_ragtruth.csv", index=False)
    print(t_rag.to_string(index=False), "\n", flush=True)
    out["ragtruth"] = t_rag

    # ------------------------------------------------- pooled ruler summary
    summary = []
    for key, label in [("mt", "MT error spans (WMT24)"),
                       ("d2t", "Data-to-text error spans"),
                       ("ragtruth", "Hallucination spans (RAGTruth)")]:
        t = out[key]
        summary.append(dict(
            scheme=label, n_systems=len(t),
            SCRIPT_min=round(t.SCRIPT.min(), 4),
            SCRIPT_max=round(t.SCRIPT.max(), 4),
            SCRIPT_median=round(float(np.median(t.SCRIPT)), 4),
            abs_z_max=round(float(t.z.abs().max()), 2),
            n_events=int(t.n_events.sum()),
            systems="; ".join(f"{r.system}={r.SCRIPT:.4f}(z={r.z:.1f})"
                              for r in t.itertuples())))
    S = pd.DataFrame(summary)
    S.drop(columns="systems").to_csv(TAB / "external_anchors.csv", index=False)
    print("== interpretation ruler ==")
    print(S.drop(columns="systems").to_string(index=False))
    print(f"\nwrote {TAB / 'external_anchors.csv'}")


if __name__ == "__main__":
    main()
