# SCRIPT

**A reference-free measure of how much of a model's behaviour was decided
before it read your message.**

📖 **[Read the illustrated explainer →](https://ADD-YOUR-USERNAME.github.io/script-metric/)**
(worked example, interactive guessing game, no maths background needed)

---

## The idea

An LLM's reply can be locally excellent and globally pre-determined. Every
sentence is appropriate, yet the same discourse moves land in the same
relative positions, in the same order, across inputs that differ in
everything that should matter.

Existing evaluation does not register this. Reference-based scores need gold
texts. LLM-as-judge grades each reply in isolation — exactly the setting where
a templated reply looks strong. Diversity statistics detect *that* outputs
converge without saying *on what*. And none of them has a chance baseline, so
their zero means nothing.

**SCRIPT** (Structural Choreography & Rigidity Index from Positions and
Transitions) turns the observation into a number. Given span annotations
alone — a label and where it starts — it reports the share of a system's
annotated-behaviour uncertainty that is explained by the reply's internal
*structure* rather than by its content, in excess of chance.

```
R = C + M

C  choreography  =  I(L ; X)            / H(L)     labels occupy fixed slots
M  momentum      =  I(L ; L_prev | X)   / H(L)     labels follow each other

SCRIPT = R_observed − mean(R_shuffled)             within-response label shuffle
z      = (R_observed − mean) / sd
```

Conditioning `M` on the position bin `X` is the load-bearing choice: without
it, momentum would double-count choreography. With it, the chain rule
guarantees `C` and `M` never count the same bit twice.

**The null is the other half of the contribution.** Raw `R` is inflated by two
artefacts: *composition* (one behaviour dominating makes the next span easy to
guess with zero structure) and *small-sample bias* (sparse count tables carry
accidental mutual information). One operation cancels both — within each
response, hold the positions fixed and randomly reassign that response's own
labels. Every label mix and every table shape is preserved, so both artefacts
hit the real and shuffled data identically. What survives is arrangement.

That gives the score a **meaningful zero**: a content-driven system reads 0 in
expectation however skewed its behaviour mix, and a system that decides its
moves before reading the input reads high however fluent each move is.

### What you get for free: the script profile

Computing `C` and `M` already builds two count tables per system — which
behaviour sits in which slot, and which behaviour follows which. Those tables
are the routine written down, and they are a **behavioural fingerprint**: the
Jensen–Shannon distance between two profiles is a behavioural distance, and
nearest-profile matching names the system that produced an unlabelled set of
annotated replies.

> The score says *how strongly* a system follows a routine.
> The profile says *which* routine — and therefore which system.

---

## SCRIPT-Seq — deciding as the annotations arrive

SCRIPT is a fixed-sample statistic: it answers *given this corpus, is the
structure real?* Three of the four uses proposed for it — a deployment
ceiling, a regression test across a release, a live monitor — are streaming
decisions instead, and a fixed-sample `z` cannot make them. Recompute `z` as
replies arrive and stop the first time it clears 1.96 and the false-positive
rate is **27%**, not 5%.

`script_betting.py` fixes that with a sequential test by betting. A gambler
bets on a payoff that is mean-zero under the null; the wealth is a
non-negative supermartingale, and Ville's inequality makes *reject when
wealth ≥ 1/α* a level-α test **simultaneously at every time step**. Betting
fractions come from a no-regret learner (Online Newton Step).

The fit is unusually clean, because SCRIPT's null already *is* an
exchangeability statement — within a reply, labels are exchangeable across
that reply's own positions:

```
at round t:  build the profile from replies 1..t-1        (past only)
             e_t = loglik(reply t) − mean over K within-reply shuffles of it
                                                          E[e_t] = 0 under H0
             W_t = W_{t-1}(1 + θ_t e_t),  θ_t by Online Newton Step
             reject when W_t ≥ 1/α
```

No reference corpus, no held-out split, no threshold to tune, no permutation
budget, and no need to fix the sample size in advance.

```bash
python script_betting.py test    spans.csv --alpha 0.05 --ci
python script_betting.py test    spans.csv --eps 0.275      # ceiling audit
python script_betting.py compare a.csv b.csv                # A is more scripted than B?
```

| what it buys | measured |
|---|---|
| false positives under the null | **0/200** streams (guarantee ≤ 0.05); naive z-peeking rejects **17/62 ≈ 27%** |
| replies to certify a frontier model | **33–41** (median), vs ~800 for the fixed-sample analysis |
| weakest template still caught every time | **50%** of the real signal |
| transfers where the score says it should | counselling certifies at t=44, AnnoMI at t=56, data-to-text at t=320; WMT24 and RAGTruth never reject |

It also yields an **anytime-valid lower bound** on the effect that tightens
reply by reply and may be watched continuously without correction — something
a fixed-sample confidence interval cannot offer.

## Use it on your own data

The metric is one self-contained file — `script_metric.py`, numpy + pandas,
no other code needed — with its own CLI.

```bash
pip install numpy pandas

python script_metric.py score spans.csv --shuffles 200 --bins 10 --profile out.json
python script_metric.py distance a.json b.json
python script_metric.py identify probe_spans.csv profiles_dir/
```

`spans.csv` needs three columns and nothing else:

| response_id | label | position |
|---|---|---|
| r001 | empathy | 0.04 |
| r001 | advice  | 0.51 |
| r001 | question| 0.92 |

`position` is the span's **start**, normalised to `[0, 1]`. (Give
`start` + `response_length` instead and it will be computed for you.) Only the
start is used, never the length — how *much* an annotator highlights is the
most annotator-dependent choice, while *where a highlight begins* is stable.

Any label scheme works: behaviour codes, error types, dialogue acts, toxicity
spans. There is no reference output, no user rating, no logits, no model
access, and no second system anywhere in the estimator.

### Reading the number

Measured on three unrelated annotation schemes, all recomputed by this
repository (`pipeline/11_external_anchors.py`):

| SCRIPT | what it looks like | measured on |
|---|---|---|
| ~0.00 | content-driven, no template | MT error spans, WMT24 |
| ~0.02 | mild structure | hallucination spans, RAGTruth |
| ~0.05 | partial structure, streaks | data-to-text error spans |
| ~0.10 | hard behavioural template | LLM counseling behaviour codes |
| 0.85  | a perfect fixed template | synthetic ground truth |

`SCRIPT = 0` means "no arrangement structure beyond chance", not "no spans".

**Operating range.** SCRIPT is biased downward below roughly 1,000 events, so
treat small strata with care and never compare a large stratum against a small
one without matching sample size first (`pipeline/09_quality_matched_check.py`
shows what that does to an apparent effect). Report `z` alongside the score:
it tells you whether the sample can support the reading.

---

## Reproducing the paper

Nothing is shipped: no data, no tables, no figures. Every input is downloaded
from its origin and every number is recomputed.

```bash
git clone <this repo> && cd script-metric
pip install -r requirements.txt
bash run_all.sh
```

Outputs land in `out/tables/` (40+ CSVs) and `out/figures/` (21 figures, PNG +
PDF). Expect a few hours end to end; the bootstrap stages are chunked and
checkpointed, so they can run in parallel or resume after an interruption.

### Data sources

`pipeline/00_get_data.py` fetches all of them, at pinned revisions, and
verifies each one before anything is allowed to run:

| Source | Where from | Checked |
|---|---|---|
| Cognitive Atrophy Benchmark (clinician span layer) | [HuggingFace `abadawi/Cognitive_Atrophy_Benchmark`](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark) | row-alignment test on every file |
| RAGTruth hallucination spans | [`ParticleMedia/RAGTruth`](https://github.com/ParticleMedia/RAGTruth) | 17,790 responses / 14,289 events |
| data-to-text error spans | [`llm-span-annotators/span-annotation`](https://github.com/llm-span-annotators/span-annotation) | 6,119 events |
| WMT24 MT error spans | same repository | 2,210 events |
| AnnoMI motivational-interviewing transcripts (SCRIPT-Seq transfer only) | [`uccollab/AnnoMI`](https://github.com/uccollab/AnnoMI) | 9,699 utterances / 4,882 therapist events |

The row-alignment test is not decoration. An earlier release of the PAIR file
had its span columns joined one row below its response columns, which silently
destroys every position in the file. `verify_alignment()` re-checks, on every
run, that sampled span texts are found in *their own* row and not better one
row up or down, and raises rather than let a shifted file through.

```bash
python pipeline/00_get_data.py                # fetch + verify everything
python pipeline/00_get_data.py --verify-only  # just re-run the checks
python pipeline/00_get_data.py benchmark      # one source
```

### Pipeline layout

Each stage is standalone and writes to `out/`. Run them in order, or run
`run_all.sh`.

```
script_metric.py               the metric — standalone, numpy + pandas, CLI
script_betting.py              SCRIPT-Seq — anytime-valid sequential testing

pipeline/
  paths.py                     paths + the 20-code scheme; no data
  fig_style.py                 shared palette

  00_get_data.py               download + verify every input corpus
  01_extract_events.py         corpora -> (label, position) events
  02_validate_metric.py        independent re-extraction; V0-V7 validation
  03_profile_suite.py          fingerprints, JS matrices, profile figures
  04_robustness_suite.py       cluster bootstrap, pairwise tests, variants
  05_sensitivity_suite.py      quality link, granularity, sample size,
                               annotator random-effects meta-analysis
  06_null_ablation_suite.py    alternative nulls, length, shuffle count
  07_loao_tier_check.py        leave-one-annotator-out tier gap
  08_multiturn_extension.py    does the script cross the turn boundary?
  09_quality_matched_check.py  the quality link at matched stratum size
  10_jury_check.py             per-annotator ("jury") scores
  11_external_anchors.py       the interpretation ruler, from source
  12_therapist_baseline.py     the human anchor, shared 3-label instrument
  13_figure_data.py            per-figure JSON for the explanatory set
  14_identification_curves.py  identification vs probe size, five settings
  15_betting_validity.py       SCRIPT-Seq: type-I error, power, sensitivity
  16_betting_applications.py   SCRIPT-Seq: calibration, ceiling audit, A/B,
                               transfer to the external corpora

  20..32_fig_*.py              one figure each
```

### Validation built into the pipeline

`02_validate_metric.py` is a from-scratch second implementation of the
extraction, not a wrapper around the first. It reports:

- **V0** its event set against `01_extract_events.py`'s — two independent
  extractions must agree (both land on 39,696 events)
- **V1/V2** per-model and per-corpus scores
- **V3** a random-label control (labels globally permuted): must read ≈ 0
- **V4** synthetic ground truth — a pure-slot corpus must put its signal in
  `C`, a pure-chain corpus in `M`, a random corpus at 0
- **V5** the profile JS matrix
- **V6** held-out-corpus identification
- **V7** bin-count and per-annotator sensitivity

`14_identification_curves.py` carries its own falsification test: the same
protocol run on globally permuted labels must land at chance (0.20). If it
does not, the protocol is leaking identity, and the script says so.
`31_fig_null_validation.py` does the same for the null itself, on corpora
whose true `R` is 0 by construction.

Several stages print an explicit warning when a published reading does *not*
hold on the recomputed numbers — for example `12_therapist_baseline.py` warns
if the human anchor is not the least-scripted speaker.

---

## Reproduction notes

Recomputing everything from source moved a handful of numbers. The
substantive claims all survive; these are the deltas worth knowing about.

**The event counts.** The pipeline extracts 39,696 span events (Qwen 8,292 /
Llama 6,108 / GPT 8,843 / Claude 8,502 / Gemini 7,951). The per-model counts
in the paper's main table are slightly lower and sum to 39,320, so that table
was produced by an earlier extraction than the one the repository ships.
SCRIPT values agree to ≤ 0.0011 either way; `z` moves more, because `z` is the
quantity most sensitive to event count.

**`z` for Claude.** The repository gives Claude `z = 56.2`. The paper's main
table says 47.7, but the Figure 2 caption independently says "≈ 56" for the
same quantity — so the two disagree with each other, and the recomputed value
matches the caption. The headline range becomes `z = 38–56`, not `z = 40–48`.

**Anchors drift in the fourth decimal.** WMT24 recomputes to −0.028–0.042
(paper: −0.030–0.043) with the outlier at 0.200 (paper: 0.196), because the
original consumed one shared RNG stream across systems while
`script_metric.compute` seeds per system. RAGTruth reproduces bit-for-bit.

**RAGTruth uses all splits, not the test split.** The published numbers match
the full 17,790-response release exactly; the test split alone leaves four of
six generators under 500 events. Earlier documentation said "test split" and
was wrong about its own numbers.

**The within-corpus identification curve is lower here** (0.33→0.63 rather
than 0.43→0.88) because this implementation enrols and probes on disjoint
replies. The held-out-corpus curve — the one the paper quotes in text —
reproduces exactly: 0.38 at k=1 to 0.73 at k=20.

**The random-label control moved sign.** The paper's table quotes +0.001
(z = 0.5); this run of the same control reads −0.003 (z = −1.8). Both are
what the control is supposed to be — indistinguishable from zero — the exact
draw depends on the permutation seed.

**Per-corpus and profile-table digits drifted with the extraction.** The
paper's per-corpus SCRIPT columns and the fingerprint table's position/share
coordinates were produced by the earlier extraction and differ from this
repository in the second or third decimal (e.g. Claude/counselchat 0.064 here
vs 0.058 in the paper; GPT empathy position 0.35 here vs 0.37). Every
qualitative reading survives: Claude is the most scripted model in all four
corpora, GPT spends half of highlighted behaviour on advice, Claude parks its
questions latest, Llama is farthest from everyone.

**Per-annotator (jury) event counts differ.** The paper describes R6 at
≈460–600 events per model and R5 at ≈210; this repository's reviewer map
gives R6 ≈260–380 and R5 ≈135. The reading is unchanged — R1–R4 significant
in all 20 jury × model cells, R6 significant for four models and null for
Llama, R5 null throughout (below the small-sample floor).

Everything else lands where the paper says it does, including: the two-tier
pairwise structure {Llama, Claude} > {Qwen, GPT, Gemini} — with the full 400
bootstrap replicates of `run_all.sh`, all six cross-tier pairs land at the
paper's Holm-adjusted p = 0.025 (Δ = 0.017–0.021) and the within-tier
p-values match to rounding (0.52 / 0.71 / 0.73 / 0.94); Claude the most
scripted model in all four corpora; the Qwen–Gemini twin pair at JS 0.016;
identification accuracy 0.556 at k=5 with the twin near-miss margin of 0.015;
the annotator meta-analysis at 0.012 ± 0.007 (z = 1.8, τ = 0.010, I² = 40%);
the leave-one-annotator-out gaps; null boundary momentum across 646 pooled
turn boundaries; sign agreement across both clinician rating axes in all
five models; the therapist as least-scripted speaker (0.079, models
0.083–0.179); the estimator variants, length terciles, granularity,
skip-gram, alternative-null, and span-length ablation tables; and the
calibration cards, bin sweep, shuffle-count, and operating-guidance numbers.

---

## Licence and citation

Code: MIT (see `LICENSE`). No dataset is redistributed; each corpus keeps its
own licence, listed in `LICENSE` and fetched by `00_get_data.py`.

The Cognitive Atrophy Benchmark is released **CC BY-NC 4.0** — non-commercial.

> Citation to be added on publication.
