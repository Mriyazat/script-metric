#!/usr/bin/env python3
"""The interpretation ruler: SCRIPT on four unrelated annotation schemes.

WMT24 MT error spans (2 labels), data-to-text error spans (6 labels), propaganda-
technique spans on human-written news (18 labels; all three from the Kasner et al.
release) and RAGTruth hallucination spans (4 labels), scored with the same estimator
and null as the benchmark. Writes tables/anchor_wmt24.csv, anchor_d2t.csv,
anchor_ragtruth.csv, anchor_propaganda.csv and the summary tables/external_anchors.csv."""
import glob
import json

import numpy as np
import pandas as pd

from pipeline.common.paths import RAGTRUTH_PATH, SPAN_ANNOTATION_DIR, TAB

from scriptmetric import metric as sm

N_BINS = 10
N_SHUFFLES = 200
SEED = 0

# Kasner et al. encode the error type as an integer; these are the released
# category names, in index order.
D2T_LABELS = ["Contradictory", "NotCheckable", "Misleading", "Incoherent",
              "Repetitive", "Other"]
MT_LABELS = ["Major", "Minor"]
# Propaganda Techniques Corpus (Da San Martino et al. 2019), as re-released by
# Kasner et al.: 18 persuasion techniques marked by trained experts on
# human-written news articles. Names in the release's config.yaml order.
PROPAGANDA_LABELS = [
    "Appeal_to_Authority", "Appeal_to_fear-prejudice", "Bandwagon",
    "Black-and-White_Fallacy", "Causal_Oversimplification", "Doubt",
    "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language",
    "Name_Calling,Labeling", "Obfuscation,Intentional_Vagueness,Confusion",
    "Red_Herring", "Reductio_ad_hitlerum", "Repetition", "Slogans",
    "Straw_Men", "Thought-terminating_Cliches", "Whataboutism"]


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


# ------------------------------------------------- propaganda techniques

def _load_propaganda(layer: str = "human") -> pd.DataFrame:
    """Propaganda-technique spans on human-written news articles.

    ``layer`` is ``"human"`` (the expert annotation) or ``"<setup>/<model>"``
    for one of the LLM-annotator layers of the release (e.g.
    ``"zeroshot/gpt4o"``). The articles are the same in every layer, so the
    LLM layers are an annotator-transfer check on identical texts. The
    "system" is the population of human authors; positions are character
    offsets over the article.
    """
    outs = {}
    for line in open(SPAN_ANNOTATION_DIR / "outputs" / "propaganda-techniques" /
                     "test.jsonl"):
        d = json.loads(line)
        outs[int(d["example_idx"])] = len(d["output"])
    fn = (SPAN_ANNOTATION_DIR / "annotations" /
          ("human/propaganda" if layer == "human" else f"model/propaganda/{layer}") /
          "annotations.jsonl")
    rows = []
    for line in open(fn):
        d = json.loads(line)
        L = outs.get(int(d["example_idx"]))
        if not L:
            continue
        for a in d["annotations"]:
            if a.get("start") is None or not (0 <= a["start"] < L):
                continue
            t = a["type"]
            lab = (PROPAGANDA_LABELS[t] if isinstance(t, int)
                   and t < len(PROPAGANDA_LABELS) else str(t))
            rows.append(dict(system="human-written news", response_id=d["example_idx"],
                             label=lab, position=a["start"] / L))
    return (pd.DataFrame(rows).sort_values(["response_id", "position"])
            .reset_index(drop=True))


def _propaganda_layers() -> list[str]:
    root = SPAN_ANNOTATION_DIR / "annotations" / "model" / "propaganda"
    return sorted(f"{p.parent.name}/{p.name}" for p in root.glob("*/*")
                  if (p / "annotations.jsonl").exists())


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

    print("== Propaganda-technique spans on human-written news ==", flush=True)
    rows = []
    prop = _load_propaganda("human")
    print(f"   {len(prop)} expert events, {prop.response_id.nunique()} articles, "
          f"{prop.label.nunique()} techniques")
    res, _ = sm.compute(prop[["response_id", "label", "position"]],
                        n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED)
    rows.append(dict(scheme="Propaganda techniques (human-written news)",
                     layer="human experts", **{k: res[k] for k in
                     ("SCRIPT", "z", "C_excess", "M_excess", "R_raw", "R_null",
                      "n_labels", "n_events", "n_responses")}))
    # random-label control: the same events, labels re-dealt within each article
    rng = np.random.default_rng(SEED + 1)
    ctrl = prop[["response_id", "label", "position"]].copy()
    ctrl["label"] = sm._shuffle_within(ctrl.label.to_numpy().copy(),
                                       pd.factorize(ctrl.response_id)[0], rng)
    res_c, _ = sm.compute(ctrl, n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED)
    rows.append(dict(scheme="Propaganda techniques (human-written news)",
                     layer="random-label control", **{k: res_c[k] for k in
                     ("SCRIPT", "z", "C_excess", "M_excess", "R_raw", "R_null",
                      "n_labels", "n_events", "n_responses")}))
    # the LLM-annotator layers of the release, on the identical articles
    for layer in _propaganda_layers():
        d = _load_propaganda(layer)
        if len(d) < 100:
            continue
        r, _ = sm.compute(d[["response_id", "label", "position"]],
                          n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED)
        rows.append(dict(scheme="Propaganda techniques (human-written news)",
                         layer=f"LLM annotator {layer}", **{k: r[k] for k in
                         ("SCRIPT", "z", "C_excess", "M_excess", "R_raw", "R_null",
                          "n_labels", "n_events", "n_responses")}))
    t_prop = pd.DataFrame(rows)
    t_prop.to_csv(TAB / "anchor_propaganda.csv", index=False)
    print(t_prop.to_string(index=False), "\n", flush=True)
    out["propaganda"] = (t_prop[t_prop.layer == "human experts"]
                         .rename(columns={"layer": "system"}))

    # ------------------------------------------------- pooled ruler summary
    summary = []
    for key, label in [("mt", "MT error spans (WMT24)"),
                       ("d2t", "Data-to-text error spans"),
                       ("ragtruth", "Hallucination spans (RAGTruth)"),
                       ("propaganda", "Propaganda techniques (human-written news)")]:
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
