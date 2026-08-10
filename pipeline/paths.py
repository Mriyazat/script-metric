"""Shared paths and constants.

Nothing here is data. Every input is downloaded from its origin by
``00_get_data.py``; every output is written under ``out/`` and is
git-ignored, so a clean checkout contains code only.
"""
import os
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
REPO = PIPELINE.parent

# make `import script_metric` work from any stage script
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ----------------------------------------------------------------- inputs
# All downloaded, never committed. Override the roots with env vars if you
# already have the corpora somewhere else.
RAW = Path(os.environ.get("SCRIPT_RAW_DIR", REPO / "raw"))
DATA_DIR = Path(os.environ.get("SCRIPT_DATA_DIR", RAW / "cognitive_atrophy"))
RAGTRUTH_PATH = Path(os.environ.get("RAGTRUTH_PATH", RAW / "ragtruth" / "response.jsonl"))
SPAN_ANNOTATION_DIR = Path(os.environ.get("SPAN_ANNOTATION_DIR", RAW / "span_annotation"))

# ---------------------------------------------------------------- outputs
OUT = Path(os.environ.get("SCRIPT_OUT_DIR", REPO / "out"))
TAB = OUT / "tables"        # every published table (CSV/JSON)
FIG = OUT / "figures"       # every published figure (PNG + PDF)
CKPT = OUT / "checkpoints"  # resumable bootstrap replicates
DERIVED = OUT / "derived"   # intermediate artefacts shared between stages
FIGDATA = DERIVED / "figdata"
EVENTS = DERIVED / "span_events.csv"   # the extracted (label, position) events

for _p in (RAW, OUT, TAB, FIG, CKPT, DERIVED, FIGDATA):
    _p.mkdir(parents=True, exist_ok=True)

# `REF` is kept as an alias so stage scripts can read artefacts that earlier
# stages derived. It is an output directory, not a shipped input.
REF = DERIVED

# -------------------------------------------------------------- constants
MODELS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
MODNUM = {"Qwen": 1, "Llama": 2, "GPT": 3, "Claude": 4, "Gemini": 5}
CORPORA = ["counselchat", "pair", "carebench", "hope"]
MULTITURN_CORPORA = ["carebench", "hope"]

# the clinician's 20-code behaviour scheme
CODES = ["SEN", "AUR", "TEN", "DIR", "FIX", "RECT", "TSH", "QOP", "QCL", "LMT",
         "MEN", "VIN", "NIN", "ASIN", "SIN", "VAC", "NAC", "ASAC", "SAC", "INC"]

# reading aid only — the metric always runs on all 20 codes
GROUPS = {
    **{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC",
                              "VIN", "NIN", "ASIN", "SIN"]},
    **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
    **{c: "questions" for c in ["QOP", "QCL", "TEN"]},
}

# column holding the human therapist's reply, per corpus
THERAPIST_COL = {"counselchat": "Therapist Response",
                 "pair": "Therapist Response (hq1)",
                 "carebench": "Original Therapist",
                 "hope": "Original Therapist"}
