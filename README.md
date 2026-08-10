# SCRIPT

**A reference-free measure of how much of a model's behaviour was decided
before it read your message.**

📖 **[Read the illustrated explainer →](https://Mriyazat.github.io/script-metric/)**
(worked example, interactive guessing game, no maths background needed)

---

## The idea

An LLM's reply can be locally excellent and globally pre-determined. Every
sentence is appropriate, yet the same discourse moves land in the same
relative positions, in the same order, across inputs that differ in
everything that should matter.

**SCRIPT** (Structural Choreography & Rigidity Index from Positions and
Transitions) turns that observation into a number. Given span annotations
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

The shuffle null gives the score a **meaningful zero**: within each response,
positions stay fixed and that response's own labels are reassigned at random,
so composition and small-sample artefacts cancel and a content-driven system
reads 0 in expectation. What survives is arrangement.

Computing `C` and `M` also builds two count tables per system — which
behaviour sits in which slot, and which follows which. Those tables are a
**behavioural fingerprint**: profiles can be compared by Jensen–Shannon
distance, and nearest-profile matching names the system that produced an
unlabelled set of annotated replies.

> The score says *how strongly* a system follows a routine.
> The profile says *which* routine — and therefore which system.

## Usage

The metric is one self-contained file — `script_metric.py`, numpy + pandas —
with its own CLI.

```bash
pip install numpy pandas

python script_metric.py score spans.csv --shuffles 200 --bins 10 --profile out.json
python script_metric.py distance a.json b.json
python script_metric.py identify probe_spans.csv profiles_dir/
```

`spans.csv` needs three columns and nothing else:

| response_id | label | position |
|---|---|---|
| r001 | VIN | 0.04 |
| r001 | DIR | 0.51 |
| r001 | QOP | 0.92 |

`position` is the span's **start**, normalised to `[0, 1]`. (Give
`start` + `response_length` instead and it will be computed for you.) Any
label scheme works: behaviour codes, error types, dialogue acts, toxicity
spans. No reference output, no user rating, no logits, no model access.

### Reading the number

| SCRIPT | what it looks like | measured on |
|---|---|---|
| ~0.00 | content-driven, no template | MT error spans, WMT24 |
| ~0.02 | mild structure | hallucination spans, RAGTruth |
| ~0.05 | partial structure, streaks | data-to-text error spans |
| ~0.05 | human counsellors' session-level routine | [AnnoMI](https://github.com/uccollab/AnnoMI) motivational interviewing |
| ~0.10 | hard behavioural template | LLM counseling behaviour codes |
| 0.85  | a perfect fixed template | synthetic ground truth |

Raw values are only comparable within one annotation scheme (the alphabet
fixes the null level), so read the rows as anchors, not a ranking. Report `z`
alongside the score, and treat strata below ~1,000 events as lower bounds.

## SCRIPT-Seq — deciding as the annotations arrive

`script_betting.py` is the sequential version: an anytime-valid test by
betting that may be checked after every reply with no correction for peeking.
It certifies a frontier-model template in ~30–40 annotated replies instead of
~800, and never rejects on the corpora where the score reads null.

```bash
python script_betting.py test    spans.csv --alpha 0.05 --ci
python script_betting.py compare a.csv b.csv     # is A more scripted than B?
```

## Reproducing the paper

Nothing is shipped: no data, no tables, no figures.

```bash
pip install -r requirements.txt
bash run_all.sh
```

`pipeline/00_get_data.py` downloads every input corpus at a pinned revision
and verifies it; the pipeline then recomputes every table and figure into
`out/`. Expect a few hours end to end.

## Licence and citation

Code: MIT (see `LICENSE`). No dataset is redistributed; each corpus keeps its
own licence, listed in `LICENSE` and fetched by `00_get_data.py`.

The Cognitive Atrophy Benchmark is released **CC BY-NC 4.0** — non-commercial.

> Citation to be added on publication.
