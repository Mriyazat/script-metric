"""Shared paths and constants.

Inputs are downloaded by ``pipeline.data.get_data`` into ``raw/``; outputs are
written under ``out/``. Both directories are git-ignored.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:                    # makes `from scriptmetric import metric` work
    sys.path.insert(0, str(REPO))

# ------------------------------------------------------------------ inputs
RAW = Path(os.environ.get("SCRIPT_RAW_DIR", REPO / "raw"))
DATA_DIR = Path(os.environ.get("SCRIPT_DATA_DIR", RAW / "cognitive_atrophy"))       # annotated CSVs
RESPONSES_DIR = RAW / "cognitive_atrophy_responses"                                 # full generated corpus
RAGTRUTH_PATH = Path(os.environ.get("RAGTRUTH_PATH", RAW / "ragtruth" / "response.jsonl"))
SPAN_ANNOTATION_DIR = Path(os.environ.get("SPAN_ANNOTATION_DIR", RAW / "span_annotation"))
ANNOMI_PATH = RAW / "annomi" / "AnnoMI-simple.csv"
MINT_DIR = Path(os.environ.get("MINT_DIR", RAW / "mint-empathy"))                  # Zhan et al. release

# ----------------------------------------------------------------- outputs
OUT = Path(os.environ.get("SCRIPT_OUT_DIR", REPO / "out"))
TAB = OUT / "tables"
TAB_B = TAB / "behaviour"          # tables of the benchmark behaviour analysis
FIG = OUT / "figures"
CKPT = OUT / "checkpoints"
DERIVED = OUT / "derived"
FIGDATA = DERIVED / "figdata"
EVENTS = DERIVED / "span_events.csv"

for _p in (RAW, OUT, TAB, TAB_B, FIG, CKPT, DERIVED, FIGDATA):
    _p.mkdir(parents=True, exist_ok=True)

REF = DERIVED                      # alias kept for the figure scripts

# --------------------------------------------------------------- constants
MODELS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
MODNUM = {"Qwen": 1, "Llama": 2, "GPT": 3, "Claude": 4, "Gemini": 5}
CORPORA = ["counselchat", "pair", "carebench", "hope"]
MULTITURN_CORPORA = ["carebench", "hope"]

CODES = ["SEN", "AUR", "TEN", "DIR", "FIX", "RECT", "TSH", "QOP", "QCL", "LMT",
         "MEN", "VIN", "NIN", "ASIN", "SIN", "VAC", "NAC", "ASAC", "SAC", "INC"]
ATTRS = ["S", "AUR", "TD", "FIX", "RT", "TN", "QOC", "LM", "ME", "EMP"]
FLAGS = ["yn_decisive", "yn_assumes", "yn_introduces", "yn_harmful", "yn_incoherent"]
USER_ATTRS = ["user_typicality", "user_evocative", "user_sensitivity", "user_request_info", "user_underlying"]

# behaviour groups (reading aid; the metric always runs on all 20 codes)
GROUPS = {
    **{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"]},
    **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
    **{c: "questions" for c in ["QOP", "QCL", "TEN"]},
}
GROUPS4 = {
    **{c: "empathy_accurate" for c in ["VAC", "NAC", "ASAC", "SAC"]},
    **{c: "empathy_inaccurate" for c in ["VIN", "NIN", "ASIN", "SIN"]},
    **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
    **{c: "questions" for c in ["QOP", "QCL", "TEN"]},
}

THERAPIST_COL = {"counselchat": "Therapist Response", "pair": "Therapist Response (hq1)",
                 "carebench": "Original Therapist", "hope": "Original Therapist"}
PROMPT_COL = {"counselchat": "prompt", "pair": "prompt", "carebench": "User Input", "hope": "User Input"}
