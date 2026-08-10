#!/usr/bin/env python3
"""Download every input corpus from its origin, then verify it.

Four sources, none of them redistributed by this repository:

  benchmark   Cognitive Atrophy Benchmark (the clinician span layer)
              -> HuggingFace  abadawi/Cognitive_Atrophy_Benchmark
  ragtruth    RAGTruth hallucination spans
              -> GitHub  ParticleMedia/RAGTruth
  spans       data-to-text and WMT24 MT error spans, from the released human
              annotations of Kasner et al., "LLMs as Span Annotators"
              -> GitHub  llm-span-annotators/span-annotation
  annomi      AnnoMI expert-annotated motivational-interviewing transcripts
              (used only by the SCRIPT-Seq transfer check, stage 16)
              -> GitHub  uccollab/AnnoMI

Revisions are pinned so a rerun gets the same bytes the paper used. Pass
``--latest`` to follow the upstream default branch instead.

    python 00_get_data.py                # all four, pinned
    python 00_get_data.py benchmark      # just one
    python 00_get_data.py --verify-only  # re-run the integrity checks

Every file is checked after download: the benchmark CSVs go through a row
alignment test (see ``verify_alignment``), and the external corpora are
checked for the exact record and event counts reported in the paper.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from paths import (CODES, CORPORA, DATA_DIR, MODNUM, RAGTRUTH_PATH, RAW,
                   SPAN_ANNOTATION_DIR)

# --------------------------------------------------------------- pinned sources

HF_REPO = "abadawi/Cognitive_Atrophy_Benchmark"
HF_REVISION = "c2dbbadaa150746f5bf666182c4f397164d0028f"   # 2026-07-09
HF_SUBDIR = "data/annotated_responses"

RAGTRUTH_REPO = "ParticleMedia/RAGTruth"
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"   # 2024-12-02

SPANS_REPO = "llm-span-annotators/span-annotation"
SPANS_REVISION = "695e033aa0fbb8a7b0031a0dbf8bdc827056b9ad"

ANNOMI_REPO = "uccollab/AnnoMI"
ANNOMI_REVISION = "42936645ec3857a9c84ab296a36a3c34b779ef49"     # 2023-03-14

# What each source must contain once downloaded. These are the counts the
# paper reports; a mismatch means the upstream release moved under us.
EXPECT_RAGTRUTH_RESPONSES = 17790
EXPECT_RAGTRUTH_EVENTS = 14289
EXPECT_D2T_EVENTS = 6119
EXPECT_MT_EVENTS = 2210
EXPECT_ANNOMI_ROWS = 9699
EXPECT_ANNOMI_THERAPIST_EVENTS = 4882


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "script-metric/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download_to(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = _get(url, timeout)
    fd, tmp = tempfile.mkstemp(dir=dest.parent)
    with open(fd, "wb") as f:
        f.write(blob)
    Path(tmp).replace(dest)


def _download_repo_subtree(repo: str, revision: str, dest: Path,
                           keep: tuple[str, ...] | None = None) -> None:
    """Fetch a GitHub archive and unpack it (optionally only some subtrees)."""
    url = f"https://github.com/{repo}/archive/{revision}.zip"
    print(f"  fetching {url}")
    blob = _get(url)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    root = zf.namelist()[0].split("/")[0]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for member in zf.namelist():
        rel = member[len(root) + 1:]
        if not rel or member.endswith("/"):
            continue
        if keep and not any(rel.startswith(k) for k in keep):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        n += 1
    print(f"  unpacked {n} files -> {dest}")


# ----------------------------------------------------------- 1. the benchmark

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", re.sub(r"[*#>_`]", "", s)).strip().lower()


def verify_alignment(path: Path, corpus: str, max_checks: int = 300) -> None:
    """Prove the annotation block is not row-shifted against the responses.

    An earlier release of the PAIR file had its span columns joined one row
    below the response columns, which silently destroys every position. The
    test: sampled span texts must be findable inside the model output of
    *their own* row, and must not match better one row up or down.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    cache: dict[tuple[int, str], str] = {}
    hits = {0: 0, 1: 0, -1: 0}
    checked = 0
    for i in range(len(df)):
        if checked >= max_checks:
            break
        for model, num in MODNUM.items():
            for code in CODES:
                col = f"Response {num}_{code}"
                if col not in df.columns:
                    continue
                v = df.iloc[i][col]
                if pd.isna(v) or not str(v).strip():
                    continue
                for span in str(v).split(" | "):
                    spn = _norm(span)
                    if len(spn) < 8:
                        continue
                    checked += 1
                    for off in (0, 1, -1):
                        j = i + off
                        if not 0 <= j < len(df):
                            continue
                        key = (j, model)
                        if key not in cache:
                            out = df.iloc[j].get(f"{model} Output")
                            cache[key] = _norm(out) if pd.notna(out) else ""
                        if spn in cache[key]:
                            hits[off] += 1
                    if checked >= max_checks:
                        break
                if checked >= max_checks:
                    break
            if checked >= max_checks:
                break
    if checked == 0:
        raise RuntimeError(f"{corpus}: found no spans to verify")
    rate = hits[0] / checked
    if rate < 0.95 or hits[0] <= max(hits[1], hits[-1]):
        raise RuntimeError(
            f"{corpus}: ROW ALIGNMENT FAILED — spans matched their own row "
            f"{hits[0]}/{checked} ({rate:.1%}) vs row+1 {hits[1]}, "
            f"row-1 {hits[-1]}. Refusing to analyse a row-shifted file.")
    print(f"  {corpus:<12} {len(df):>4} rows  alignment OK "
          f"({hits[0]}/{checked} = {rate:.1%} own-row)")


def get_benchmark(revision: str) -> None:
    print(f"[benchmark] {HF_REPO} @ {revision[:12]}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for corpus in CORPORA:
        fname = f"{corpus}_annotated.csv"
        url = (f"https://huggingface.co/datasets/{HF_REPO}/resolve/"
               f"{revision}/{HF_SUBDIR}/{fname}")
        dest = DATA_DIR / fname
        if not dest.exists():
            print(f"  fetching {fname}")
            _download_to(url, dest)
    verify_benchmark()


def verify_benchmark() -> None:
    for corpus in CORPORA:
        p = DATA_DIR / f"{corpus}_annotated.csv"
        if not p.exists():
            raise RuntimeError(f"missing {p} — run `00_get_data.py benchmark`")
        verify_alignment(p, corpus)


# ------------------------------------------------------------- 2. RAGTruth

def get_ragtruth(revision: str) -> None:
    print(f"[ragtruth] {RAGTRUTH_REPO} @ {revision}")
    dest = RAGTRUTH_PATH.parent
    if not RAGTRUTH_PATH.exists():
        _download_repo_subtree(RAGTRUTH_REPO, revision, dest, keep=("dataset/",))
        # flatten dataset/response.jsonl -> response.jsonl
        inner = dest / "dataset" / "response.jsonl"
        if inner.exists():
            inner.replace(RAGTRUTH_PATH)
            src_info = dest / "dataset" / "source_info.jsonl"
            if src_info.exists():
                src_info.replace(dest / "source_info.jsonl")
            shutil.rmtree(dest / "dataset", ignore_errors=True)
    verify_ragtruth()


def verify_ragtruth() -> None:
    if not RAGTRUTH_PATH.exists():
        raise RuntimeError(f"missing {RAGTRUTH_PATH} — run `00_get_data.py ragtruth`")
    recs = [json.loads(l) for l in open(RAGTRUTH_PATH)]
    events = sum(len(r.get("labels") or []) for r in recs)
    labelled = sum(1 for r in recs if r.get("labels"))
    gens = sorted({r["model"] for r in recs})
    if len(recs) != EXPECT_RAGTRUTH_RESPONSES or events != EXPECT_RAGTRUTH_EVENTS:
        raise RuntimeError(
            f"RAGTruth changed upstream: got {len(recs)} responses / {events} "
            f"events, expected {EXPECT_RAGTRUTH_RESPONSES} / "
            f"{EXPECT_RAGTRUTH_EVENTS}. Re-check the anchor before publishing.")
    print(f"  {len(recs)} responses, {labelled} with spans, {events} events, "
          f"{len(gens)} generators — OK")


# --------------------------- 3. data-to-text + WMT24 error spans (Kasner et al.)

def get_span_annotation(revision: str) -> None:
    print(f"[spans] {SPANS_REPO} @ {revision[:12]}")
    marker = SPAN_ANNOTATION_DIR / "annotations" / "human" / "mt-eval" / "annotations.jsonl"
    if not marker.exists():
        _download_repo_subtree(SPANS_REPO, revision, SPAN_ANNOTATION_DIR,
                               keep=("annotations/human/", "outputs/", "LICENSE",
                                     "README.md"))
    verify_span_annotation()


def _count_span_events(kind: str) -> int:
    """Count usable (label, position) events exactly as 11_external_anchors does."""
    import glob
    lengths = {}
    for f in glob.glob(str(SPAN_ANNOTATION_DIR / "outputs" / kind / "*" / "*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            lengths[(d["dataset"], d["split"], int(d["example_idx"]),
                     d["setup_id"])] = len(d["output"])
    ann_files = ([SPAN_ANNOTATION_DIR / "annotations" / "human" / kind / s /
                  "annotations.jsonl" for s in ("test", "dev")]
                 if kind == "d2t-eval" else
                 [SPAN_ANNOTATION_DIR / "annotations" / "human" / kind /
                  "annotations.jsonl"])
    n = 0
    for fn in ann_files:
        if not fn.exists():
            continue
        for line in open(fn):
            d = json.loads(line)
            L = lengths.get((d["dataset"], d["split"], int(d["example_idx"]),
                             d["setup_id"]))
            if not L:
                continue
            n += sum(1 for a in d["annotations"]
                     if a["start"] is not None and a["start"] < L)
    return n


def verify_span_annotation() -> None:
    marker = SPAN_ANNOTATION_DIR / "annotations" / "human" / "mt-eval" / "annotations.jsonl"
    if not marker.exists():
        raise RuntimeError(f"missing {marker} — run `00_get_data.py spans`")
    d2t, mt = _count_span_events("d2t-eval"), _count_span_events("mt-eval")
    if d2t != EXPECT_D2T_EVENTS or mt != EXPECT_MT_EVENTS:
        raise RuntimeError(
            f"span-annotation changed upstream: d2t {d2t} (expected "
            f"{EXPECT_D2T_EVENTS}), mt {mt} (expected {EXPECT_MT_EVENTS}).")
    print(f"  data-to-text {d2t} events, WMT24 {mt} events — OK")


# -------------------------------------------------- 4. AnnoMI (optional, stage 16)

def get_annomi(revision: str) -> None:
    print(f"[annomi] {ANNOMI_REPO} @ {revision[:12]}")
    dest = RAW / "annomi"
    if not (dest / "AnnoMI-simple.csv").exists():
        url = f"https://github.com/{ANNOMI_REPO}/archive/{revision}.zip"
        print(f"  fetching {url}")
        zf = zipfile.ZipFile(io.BytesIO(_get(url)))
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for member in zf.namelist():
            if member.endswith(("AnnoMI-simple.csv", "AnnoMI-full.csv")):
                with zf.open(member) as src, \
                        open(dest / Path(member).name, "wb") as out:
                    shutil.copyfileobj(src, out)
                n += 1
        print(f"  unpacked {n} files -> {dest}")
    verify_annomi()


def verify_annomi() -> None:
    p = RAW / "annomi" / "AnnoMI-simple.csv"
    if not p.exists():
        raise RuntimeError(f"missing {p} — run `00_get_data.py annomi`")
    d = pd.read_csv(p)
    events = d[(d.interlocutor == "therapist")
               & d.main_therapist_behaviour.notna()].shape[0]
    if len(d) != EXPECT_ANNOMI_ROWS or events != EXPECT_ANNOMI_THERAPIST_EVENTS:
        raise RuntimeError(
            f"AnnoMI changed upstream: got {len(d)} rows / {events} therapist "
            f"events, expected {EXPECT_ANNOMI_ROWS} / "
            f"{EXPECT_ANNOMI_THERAPIST_EVENTS}.")
    print(f"  {len(d)} utterances, {d.transcript_id.nunique()} transcripts, "
          f"{events} annotated therapist events — OK")


# ------------------------------------------------------------------------ CLI

SOURCES = {
    "benchmark": (get_benchmark, verify_benchmark, HF_REVISION, "main"),
    "ragtruth": (get_ragtruth, verify_ragtruth, RAGTRUTH_REVISION, "main"),
    "spans": (get_span_annotation, verify_span_annotation, SPANS_REVISION, "main"),
    "annomi": (get_annomi, verify_annomi, ANNOMI_REVISION, "main"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", metavar="SOURCE",
                    help=f"which of {', '.join(SOURCES)} to fetch (default: all)")
    ap.add_argument("--latest", action="store_true",
                    help="follow the upstream default branch instead of the pin")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip downloading; just re-run the integrity checks")
    args = ap.parse_args()

    wanted = args.sources or list(SOURCES)
    unknown = [s for s in wanted if s not in SOURCES]
    if unknown:
        sys.exit(f"unknown source(s) {unknown}; choose from {list(SOURCES)}")
    print(f"raw data root: {RAW}\n")
    for name in wanted:
        fetch, verify, pinned, latest = SOURCES[name]
        try:
            if args.verify_only:
                print(f"[{name}] verify only")
                verify()
            else:
                fetch(latest if args.latest else pinned)
        except urllib.error.URLError as e:
            sys.exit(f"\n{name}: download failed ({e}). Check your network, or "
                     f"place the files under {RAW} by hand.")
        print()
    print("all requested sources present and verified.")


if __name__ == "__main__":
    main()
