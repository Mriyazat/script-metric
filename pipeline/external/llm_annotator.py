#!/usr/bin/env python3
"""Does SCRIPT survive replacing the clinicians with a blind LLM annotator?

Every reply (optionally the human therapist's as well) is re-annotated under the same
20-code scheme by an LLM that sees only the user message and one reply, in shuffled
order. Events are built from the returned verbatim quotes exactly as for the
clinician layer, and SCRIPT is compared against the clinician layer on the same replies.
Results are cached per reply, so interrupted runs resume.

    DEEPSEEK_API_KEY=...  python -m pipeline.external.llm_annotator --annotator deepseek-v4-pro --include-therapist
    OPENAI_API_KEY=...    python -m pipeline.external.llm_annotator --annotator gpt-4.1-mini
    ANTHROPIC_API_KEY=... python -m pipeline.external.llm_annotator --annotator claude-sonnet-4-5

    --rows N     annotate a random N of the 819 prompt rows;  --dry-run  print one prompt

Writes out/derived/llm_span_events.csv and tables/llm_annotator_comparison.csv."""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pipeline.common.paths import (CODES, CORPORA, DATA_DIR, DERIVED, EVENTS, MODELS, TAB,
                   THERAPIST_COL)

from pipeline.common.benchmark import norm
from scriptmetric.metric import compute

CACHE = DERIVED / "llm_annotator"

# The 20-code scheme as shown to the clinician annotators (names from the
# annotation UI; definitions from the coding manual as summarised in the
# paper's attribute-span map).
CODEBOOK = """\
SEN   Sensitivity: sensitive or safety-related content (risk, crisis, harm).
AUR   Assumption of User Response: treats the user's account as accurate \
without qualification.
TEN   Tentativeness: tentative, exploratory, hedged phrasing.
DIR   Directness: directive or prescriptive phrasing (telling the user what \
to do or what is true).
FIX   Fix It: attempts to solve or resolve the user's concern.
RECT  Recommendation Type: a concrete recommendation, either skill-based \
(e.g. breathing exercise) or action-oriented (e.g. see a therapist).
TSH   Topic Shift: moves away from the user's expressed topic.
QOP   Open Question: question inviting an open-ended answer.
QCL   Closed Question: question inviting a yes/no or short factual answer.
LMT   Language Matching: reuses the user's own wording or framing.
MEN   Minimal Encouragers: brief encouragers such as "mhm", "aha", "wow".
VAC   Accurate Validation: validates feelings the user actually expressed.
VIN   Inaccurate Validation: validates feelings the user did not express, \
or validates inaccurately.
NAC   Accurate Normalizing: normalizes an experience the user described.
NIN   Inaccurate Normalizing: normalizes something the user did not \
describe, or normalizes inappropriately.
ASAC  Accurate Autonomy Support: supports the user's own agency or choice, \
grounded in what they said.
ASIN  Inaccurate Autonomy Support: autonomy-supportive phrasing not \
grounded in what the user said.
SAC   Accurate Support: supportive or encouraging statement grounded in \
the user's account.
SIN   Inaccurate Support: supportive statement not grounded in the user's \
account.
INC   Incoherent / Other: incoherent fragment, or behaviour outside the \
codes above."""

SYSTEM = """\
You are a clinician-trained annotator coding counseling-style replies with \
a fixed 20-code span scheme. You will see the user's message for context \
and one assistant reply. Mark every span of the REPLY that instantiates \
one of the codes.

Codes:
{codebook}

Rules:
- Annotate the reply only, never the user's message.
- A span is a contiguous verbatim quote from the reply, usually a clause \
or sentence, at most ~40 words.
- Mark every instance, in order of appearance. The same sentence may \
receive more than one code (return it once per code).
- Copy quotes EXACTLY, character for character, from the reply.
- If nothing applies, return an empty list.

Return ONLY json, no prose, of the form:
{{"spans": [{{"code": "VAC", "quote": "exact text from the reply"}}, ...]}}"""

USER = """\
USER MESSAGE (context only, do not annotate):
{prompt}

REPLY TO ANNOTATE:
{reply}"""

PROMPT_COL = {"counselchat": "prompt", "pair": "prompt",
              "carebench": "User Input", "hope": "User Input"}


# --------------------------------------------------------------- providers

def call_openai_compatible(model, system, user, base, key, timeout=180,
                           deepseek=False):
    body = {"model": model, "temperature": 0, "max_tokens": 6000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if deepseek:
        # thinking mode burns the whole token budget on reasoning and
        # returns empty content; span extraction doesn't need it
        body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ch = json.load(r)["choices"][0]
    if ch.get("finish_reason") == "length":
        return None                       # truncated: caller treats as failure
    return ch["message"]["content"]


PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
               "OPENAI_API_KEY"),
}


def call_anthropic(model, system, user, timeout=180):
    body = {"model": model, "max_tokens": 4096, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["content"][0]["text"]


def call_llm(provider, model, system, user, tries=4):
    for k in range(tries):
        try:
            if provider == "anthropic":
                return call_anthropic(model, system, user)
            base, key_env = PROVIDERS[provider]
            return call_openai_compatible(model, system, user, base,
                                          os.environ[key_env],
                                          deepseek=provider == "deepseek")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and k < tries - 1:
                time.sleep(5 * 2 ** k)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if k < tries - 1:
                time.sleep(5 * 2 ** k)
                continue
            raise


def parse_spans(text):
    """Tolerant JSON extraction: accept {"spans": [...]} or a bare array.

    Returns None (= failure, retry) if the text is empty or contains no
    JSON at all; an empty span list is only accepted from valid JSON.
    """
    if not text:
        return None
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(arr, dict):
        arr = arr.get("spans", [])
    if not isinstance(arr, list):
        return None
    out = []
    for o in arr:
        if isinstance(o, dict) and str(o.get("code", "")).upper() in CODES \
                and str(o.get("quote", "")).strip():
            out.append({"code": str(o["code"]).upper(),
                        "quote": str(o["quote"])})
    return out


# ------------------------------------------------------------------- data

def reply_table(include_therapist=False):
    rows = []
    speakers = MODELS + (["Human"] if include_therapist else [])
    for corpus in CORPORA:
        df = pd.read_csv(DATA_DIR / f"{corpus}_annotated.csv", low_memory=False)
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        for i, r in df.iterrows():
            prompt = str(r.get(PROMPT_COL[corpus], "") or "")
            for m in speakers:
                col = THERAPIST_COL[corpus] if m == "Human" else f"{m} Output"
                out = r.get(col)
                if pd.isna(out) or not str(out).strip():
                    continue
                rows.append(dict(corpus=corpus, row=i, model=m,
                                 prompt=prompt, reply=str(out)))
    return pd.DataFrame(rows)


def _annotate_one(r, provider, annotator, system, cache_dir):
    f = cache_dir / f"{r.corpus}_{r.row}_{r.model}.json"
    if f.exists():
        return 0
    user = USER.format(prompt=r.prompt, reply=r.reply)
    spans = parse_spans(call_llm(provider, annotator, system, user))
    if spans is None:                   # empty/truncated/malformed: one retry
        spans = parse_spans(call_llm(provider, annotator, system, user))
    if spans is None:                   # do not cache failures; rerun retries
        return 1
    f.write_text(json.dumps({"spans": spans}, ensure_ascii=False))
    return 0


def annotate(T, provider, annotator, cache_dir, workers=8):
    cache_dir.mkdir(parents=True, exist_ok=True)
    system = SYSTEM.format(codebook=CODEBOOK)
    rows = list(T.itertuples())
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_annotate_one, r, provider, annotator, system,
                          cache_dir) for r in rows]
        for fut in as_completed(futs):
            failed += fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  annotated {done}/{len(rows)}", flush=True)
    if failed:
        print(f"  WARNING: {failed} replies returned unparseable JSON")


def build_events(T, cache_dir):
    rows, n_total, n_found = [], 0, 0
    for _, r in T.iterrows():
        f = cache_dir / f"{r.corpus}_{r.row}_{r.model}.json"
        if not f.exists():
            continue
        txt = norm(r.reply)
        L = len(txt)
        if L == 0:
            continue
        seen = set()
        for sp in json.loads(f.read_text())["spans"]:
            spn = norm(sp["quote"])
            if not spn:
                continue
            n_total += 1
            p = txt.find(spn)
            if p < 0:
                continue
            n_found += 1
            key = (sp["code"], p)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(corpus=r.corpus, row=r.row, model=r.model,
                             label=sp["code"], position=p / L))
    E = pd.DataFrame(rows)
    pct = 100 * n_found / n_total if n_total else 0
    print(f"LLM spans: searched {n_total}, located {n_found} ({pct:.1f}%), "
          f"events after dedupe {len(E)}")
    return E


def score_by_model(E, speakers):
    out = {}
    for m in speakers:
        d = E[E.model == m].copy()
        if len(d) < 50:
            continue
        d["response_id"] = (d.corpus.astype(str) + "|" + d.row.astype(str)
                            + "|" + d.model.astype(str))
        d = d.sort_values(["response_id", "position"]).reset_index(drop=True)
        res, _ = compute(d[["response_id", "label", "position"]], seed=0)
        out[m] = res
    return out


def random_label_control(E, seed=0):
    """The LLM's own events, labels shuffled within each reply."""
    d = E.copy()
    d["response_id"] = (d.corpus.astype(str) + "|" + d.row.astype(str)
                        + "|" + d.model.astype(str))
    d = d.sort_values(["response_id", "position"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    d["label"] = (d.groupby("response_id")["label"]
                    .transform(lambda s: rng.permutation(s.to_numpy())))
    res, _ = compute(d[["response_id", "label", "position"]], seed=0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", help="annotator model id (e.g. deepseek-v4-pro)")
    ap.add_argument("--provider", choices=["deepseek", "openai", "anthropic"],
                    help="default: inferred from which API key is set")
    ap.add_argument("--rows", type=int, default=0,
                    help="sample this many prompt rows (0 = all 819)")
    ap.add_argument("--include-therapist", action="store_true",
                    help="send the human therapist replies through the same "
                         "blind pipeline")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild the event layer and the comparison table from the "
                         "per-reply cache only; no API key and no network access")
    args = ap.parse_args()

    T = reply_table(args.include_therapist)
    if args.rows:
        keys = sorted({(c, r) for c, r in zip(T.corpus, T.row)})
        random.Random(args.seed).shuffle(keys)
        keep = set(keys[:args.rows])
        T = T[[k in keep for k in zip(T.corpus, T.row)]].reset_index(drop=True)
    print(f"{len(T)} replies ({T.groupby('model').size().to_dict()})")

    if args.dry_run:
        r = T.iloc[0]
        print("\n----- SYSTEM -----\n" + SYSTEM.format(codebook=CODEBOOK))
        print("\n----- USER -----\n" + USER.format(prompt=r.prompt, reply=r.reply))
        return

    if not args.annotator:
        sys.exit("--annotator is required (or use --dry-run)")
    cache_dir = CACHE / re.sub(r"[^\w.-]", "_", args.annotator)
    # shuffled call order: no positional cue about speaker or corpus
    T = T.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.from_cache:
        n_cached = sum((cache_dir / f"{r.corpus}_{r.row}_{r.model}.json").exists()
                       for r in T.itertuples())
        print(f"rebuilding from cache {cache_dir}: {n_cached}/{len(T)} replies cached")
    else:
        provider = args.provider or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else
            "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
            "openai" if os.environ.get("OPENAI_API_KEY") else None)
        if provider is None:
            sys.exit("set DEEPSEEK_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print(f"annotating with {args.annotator} via {provider}; cache {cache_dir}")
        annotate(T, provider, args.annotator, cache_dir, workers=args.workers)

    E = build_events(T, cache_dir)
    E.to_csv(DERIVED / "llm_span_events.csv", index=False)
    speakers = MODELS + (["Human"] if args.include_therapist else [])
    llm = score_by_model(E, speakers)

    # clinician events restricted to the identical reply subset
    C = pd.read_csv(EVENTS)
    keys = set(zip(E.corpus, E.row, E.model))
    C = C[[k in keys for k in zip(C.corpus, C.row, C.model)]]
    clin = score_by_model(C, MODELS)

    rows = []
    for m in speakers:
        if m not in llm:
            continue
        cl = clin.get(m, {})
        rows.append(dict(model=m,
                         SCRIPT_llm=llm[m]["SCRIPT"], z_llm=llm[m]["z"],
                         C_llm=llm[m]["C_excess"], M_llm=llm[m]["M_excess"],
                         n_events_llm=llm[m]["n_events"],
                         SCRIPT_clin=cl.get("SCRIPT"), z_clin=cl.get("z"),
                         C_clin=cl.get("C_excess"), M_clin=cl.get("M_excess"),
                         n_events_clin=cl.get("n_events")))
    ctrl = random_label_control(E[E.model.isin(MODELS)], seed=args.seed)
    rows.append(dict(model="random-label control (LLM events)",
                     SCRIPT_llm=ctrl["SCRIPT"], z_llm=ctrl["z"],
                     C_llm=ctrl["C_excess"], M_llm=ctrl["M_excess"],
                     n_events_llm=ctrl["n_events"]))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "llm_annotator_comparison.csv", index=False)
    print(R.to_string(index=False))
    S = R[R.model.isin(MODELS)].dropna(subset=["SCRIPT_clin"])
    if len(S) >= 3:
        rho = S.SCRIPT_llm.rank().corr(S.SCRIPT_clin.rank())
        print(f"\nrank agreement (Spearman) LLM vs clinician: {rho:.2f}")
    print(f"wrote {TAB / 'llm_annotator_comparison.csv'}")


if __name__ == "__main__":
    main()
