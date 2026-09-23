<div align="center">

<br>

<h1>
  <img src="https://img.shields.io/badge/-SCRIPT-26323a?style=for-the-badge&labelColor=26323a&color=26323a" height="46" alt="SCRIPT">
</h1>

<h3>Measuring Behavioural Scriptedness from Span Annotations</h3>

<p>
<b>Do language models listen, or do they run a blueprint?</b><br>
SCRIPT reads only the <i>(behaviour, position)</i> events that annotated evaluations already produce and asks how much of what a system does is fixed by <i>where it sits in the response</i> rather than by <i>what the user said</i>.
</p>

<p>
<img src="https://img.shields.io/badge/python-3.10%2B-3d6fae?style=flat-square&logo=python&logoColor=white" alt="python">
<img src="https://img.shields.io/badge/deps-numpy%20%C2%B7%20pandas-8e5bb5?style=flat-square" alt="deps">
<img src="https://img.shields.io/badge/reproduce-one%20command%20per%20table-2a9d8f?style=flat-square" alt="reproduce">
<img src="https://img.shields.io/badge/code-MIT-4f9d6b?style=flat-square" alt="license">
</p>

<br>

<img src="assets/figure1_method.png" width="100%" alt="Figure 1: overview of SCRIPT">

<p><sub><b>Figure 1 of the paper.</b> <b>(a)</b> A span-annotated response. <b>(b)</b> Each span becomes an event: its label and the position bin where it starts. <b>(c)</b> Pooled over a system's responses, the events fill a position table and a transition table, which give choreography <i>C</i> and momentum <i>M</i>. <b>(d)</b> Within each response the labels are re-dealt over the fixed positions; SCRIPT is the observed rigidity in excess of the shuffle mean. <b>(e)</b> The two tables are the system's script profile; scoring a new response under each enrolled profile names the system that produced it.</sub></p>

</div>

<br>

<table align="center" width="100%">
<tr>
<td width="33%" valign="top">
<h4 align="center">Meaningful zero</h4>
<p align="center"><sub>Within each response the labels are re-dealt over the fixed positions, so the behaviour mix is unchanged and only the arrangement is destroyed. A system driven purely by content reads <b>0</b>, whatever its mix or sample size.</sub></p>
</td>
<td width="33%" valign="top">
<h4 align="center">Seats and chains</h4>
<p align="center"><sub>The score splits exactly into <b>choreography</b> (behaviours in fixed positions) and <b>momentum</b> (behaviours chaining into one another), reported in one unit across annotation schemes.</sub></p>
</td>
<td width="33%" valign="top">
<h4 align="center">Whose script</h4>
<p align="center"><sub>The same count tables form a <b>script profile</b> that names the generating model from a handful of annotated responses, and a sequential test that certifies a blueprint as annotations arrive.</sub></p>
</td>
</tr>
</table>

<br>

<div align="center">

**[Quick start](#quick-start) · [Main text, one command each](#main-text-one-command-each) · [Appendix](#appendix-one-command-per-section) · [Your own data](#run-script-on-your-own-annotations) · [Data](#data) · [Layout](#layout) · [Cite](#citation-and-licence)**

</div>

<br>

## Quick start

```bash
git clone <this repository> script-metric && cd script-metric
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash reproduce.sh data      # fetch and verify every input at the pinned revision (~500 MB, once)
bash reproduce.sh table2    # any table or figure of the paper, by name
bash reproduce.sh list      # the full list of targets
```

Python 3.10+. The metric needs `numpy` and `pandas`; the paper's pipeline adds `matplotlib` and `scipy`. Every target runs **without an API key**: the blind LLM re-annotation of Testbed 1 ships as a per-reply cache under `out/derived/llm_annotator/`.

<details>
<summary><b>Optional extras</b> (two hand-built figures; Zhan et al.'s sentence split)</summary>
<br>

- **Figure 1 and the profile figure** are built from HTML (`pipeline/figures/handmade/`) and exported with a headless browser: `pip install playwright`; it uses your installed Google Chrome, or run `python -m playwright install chromium`.
- **Testbed 2's sentence split** follows Zhan et al. with `nltk` + `punkt`: `pip install nltk && python -c "import nltk; nltk.download('punkt_tab')"`. Without it a regex splitter is used and a few sentence counts differ.
- **Regenerating the LLM layer from scratch** (a few dollars of API time): `DEEPSEEK_API_KEY=… python -m pipeline.external.llm_annotator --annotator deepseek-v4-pro --include-therapist`.

</details>

<br>

## Main text, one command each

Every table and figure of Section 5 has a target. Each prints the files it wrote; open them beside the paper.

```bash
bash reproduce.sh main      # Tables 2–4, Figures 2–3 and every quoted number, in order (≈ 1–2 h)
```

| | Paper item | Command | Output |
|:-:|---|---|---|
| **T2** | **One level, one choreography, and the human anchor.** SCRIPT, *C*, *M*, *z* for five models under the clinician and the blind-LLM annotator; paired Δ*C* against the therapist; random-label control | `bash reproduce.sh table2` | `out/derived/validation_results.csv` · `out/tables/llm_annotator_comparison.csv` · `therapist_paired_C.csv` · `pairwise_tests.csv` · `out/verify_report.json` |
| **T3** | **Does the behaviour answer the person or the conversation?** Advice shares by affect and by turn; shifts on affect and safety cue; turn, person and seat shares of *H(L)* | `bash reproduce.sh table3` | `out/tables/listening_arc.csv` · `listening_user_state.csv` · `listening_budget.csv` |
| **T4** | **What the judge sees and what SCRIPT sees.** Matched-mix tests (arrangement, length control, Testbed 1 clinicians) and the 4B ladder of Testbed 2 | `bash reproduce.sh table4` | `out/tables/conformity_tb2_pairs_own.csv` · `conformity_tb1_pairs_own.csv` · `conformity_length_control.csv` · `mint_systems.csv` · `mint_surface_layer.csv` |
| **F1** | **Overview of SCRIPT** (hand-built) | `bash reproduce.sh fig1` | `pipeline/figures/handmade/method_figure/script_method_figure.png` |
| **F2** | **The blueprint, turn by turn, and what moves it.** (a) the behaviour dominating each seat at each turn, six speakers; (b) each model's shift with a user cue relative to the therapist's | `bash reproduce.sh fig2` | `out/figures/fig_results_main.png` |
| **F3** | **The profile, the ruler, and the sequential test.** (a) naming the model from *k* responses, with baselines; (b) SCRIPT for every corpus in the paper; (c) SCRIPT-Seq wealth | `bash reproduce.sh fig3` | `out/figures/fig_results_closing.png` · `out/tables/identification_curves.csv` · `identification_baselines.csv` |
| **#** | **Every number quoted in §5.1–5.4** | `bash reproduce.sh numbers` | `out/tables/latex/numbers.json` and a LaTeX body per table in `out/tables/latex/` |

Table 1 (the two testbeds) is descriptive: its counts are the row totals of `out/derived/span_events.csv` and `out/tables/mint_systems.csv`. `bash reproduce.sh figures ../figures` places the finished figures into a LaTeX folder.

<br>

## Appendix, one command per section

```bash
bash reproduce.sh appendix  # everything below (several hours; the sensitivity sweeps and SCRIPT-Seq streams dominate)
```

<details open>
<summary><b>A · Testbed 1: codebook, corpora, annotation layers, and anchors</b></summary>
<br>

| Section | Command | Output |
|---|---|---|
| A · agreement of the blind LLM layer with the clinicians; anchor corpora | `bash reproduce.sh A` | `llm_agreement*.csv` · `external_anchors.csv` |

</details>

<details open>
<summary><b>B · Results in full</b> — one subsection per main-text subsection</summary>
<br>

| Section | Command | Output |
|---|---|---|
| B.1 · One blueprint across models; humans read lower — therapist ceiling fraction, density bands, the six human-reference settings | `bash reproduce.sh B1` | `therapist_llm_ceiling.csv` · `therapist_density_matched.csv` · `therapist_baseline.csv` · `anchor_empathy_tactics.csv` |
| B.2 · Does the blueprint answer the person? — full listening tables, density-matched shares, seat strength, cross-turn structure | `bash reproduce.sh B2` | `listening_*.csv` · `seat_strength.csv` · `script_by_turn.csv` · `multiturn_extension.csv` · `fig_multiturn.png` |
| B.3 · Existing evaluation does not see the blueprint — all 22 Testbed-2 systems, the hand-written template under the null, stickiness decomposed, human anchors under the ten-tactic scheme | `bash reproduce.sh B3` | `mint_systems.csv` · `regex_under_null.csv` · `stickiness_decomposed.csv` · `anchor_empathy_tactics.csv` |
| B.4 · The profile names the model — identification baselines, JS distances, twins per corpus, case study, profile figure | `bash reproduce.sh B4` | `identification_baselines.csv` · `twin_structure_by_corpus.csv` · `fig_case_study.png` · `fig_transition_lift.png` · `script_profile_figure.png` |
| B.5 · The level survives the annotator, the estimator, and the task — bootstrap intervals, alternative nulls, length terciles, annotator juries, propaganda anchor | `bash reproduce.sh B5` | `bootstrap_ci.csv` · `alternative_nulls.csv` · `length_terciles.csv` · `jury_scripts.csv` · `loao_tier_gap.csv` · `anchor_propaganda.csv` |

</details>

<details open>
<summary><b>C · The estimator in detail</b></summary>
<br>

| Section | Command | Output |
|---|---|---|
| C.1 · Transition conventions for the momentum term | `bash reproduce.sh C1` | `tie_robustness.csv` · `fig_momentum.png` |
| C.2 · Validation of the null calibration | `bash reproduce.sh C2` | `null_validation.csv` · `fig_null_validation.png` · `fig_anatomy.png` |
| C.3 · Sensitivity of the estimator and of the extensions | `bash reproduce.sh C3` | `bin_sweep.csv` · `granularity.csv` · `skipgram.csv` · `reviewer_qs/*.csv` |
| C.4 · The script profile: construction and matching | `bash reproduce.sh C4` | `fig_worked_example.png` · `fig_method_faithful.png` |
| C.5 · SCRIPT-Seq: the sequential test in full | `bash reproduce.sh C5` | `betting_validity.csv` · `reviewer_qs/q7_*.csv` · `betting_transfer.csv` · `fig_betting.png` · `fig_evidence_ladder.png` |
| C.6 · Per-response conformity to the blueprint and rated quality | `bash reproduce.sh C6` | `conformity_*.csv` · `conformity_summary.json` · `quality_matched.csv` |

</details>

<details open>
<summary><b>D · Testbed 1's behaviour, described</b></summary>
<br>

| Section | Command | Output |
|---|---|---|
| D · attributes, spans, therapist comparison, turn dynamics, repeated phrase families | `bash reproduce.sh D` | `out/tables/behaviour/*.csv` · `fig_behaviour_overview.png` · `fig_behaviour_language.png` |

</details>

Paths without a folder are under `out/tables/` (tables) or `out/figures/` (figures). `bash run_all.sh` is the same pipeline as one script, in dependency order, with checkpointed bootstraps.

<br>

## Run SCRIPT on your own annotations

The metric is one module, `scriptmetric/metric.py`, importable (`from scriptmetric import metric`) and runnable from the command line. Input is a CSV of spans with three columns, `response_id`, `label`, `position` (the span's start normalised to [0, 1]; or give `start` and `response_length`). Any label scheme works: behaviour codes, error types, dialogue acts, hallucination spans.

```bash
python -m scriptmetric.metric  score    spans.csv --shuffles 200 --bins 10 --profile out.json
python -m scriptmetric.metric  score    spans.csv --ceiling            # fraction of achievable rigidity
python -m scriptmetric.metric  distance a.json b.json                  # how far apart are two systems' profiles?
python -m scriptmetric.metric  identify probe_spans.csv profiles_dir/  # which enrolled system wrote these?
```

`score` returns SCRIPT, its split into choreography *C* and momentum *M*, and *z* against the within-response shuffle null. `scriptmetric/betting.py` is SCRIPT-Seq, the anytime-valid version for annotations that arrive over time:

```bash
python -m scriptmetric.betting test    spans.csv --alpha 0.05 --ci   # certify a blueprint; valid at any stopping time
python -m scriptmetric.betting compare a.csv b.csv                   # is A more scripted than B?
```

<br>

## Data

Nothing is redistributed. `reproduce.sh data` fetches each source from its origin at a pinned revision and verifies it.

| Source | Role in the paper | Origin |
|---|---|---|
| Cognitive Atrophy Benchmark | **Testbed 1** — 819 counseling prompts, five model replies each, clinician scores, flags and highlighted spans, the therapist's reply | [HF abadawi/Cognitive_Atrophy_Benchmark](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark) |
| MINT / Lend an Ear (Zhan et al.) | **Testbed 2** — 22 system outputs on 315 turns with tactic tags and expert-protocol ratings; 322 tagged real conversations | [honglizhan/mint-empathy](https://github.com/honglizhan/mint-empathy) |
| Gueorguieva et al., Study 1 | human-coded empathy tactics on Reddit support responses (human writers, GPT-4-turbo, Llama-3-70B) | released with the above |
| AnnoMI | human motivational-interviewing counsellors (human anchor) | [uccollab/AnnoMI](https://github.com/uccollab/AnnoMI) |
| Kasner et al. span annotations | WMT24 MT error spans (expected null), data-to-text error spans, propaganda-technique spans on human news | [llm-span-annotators/span-annotation](https://github.com/llm-span-annotators/span-annotation) |
| RAGTruth | hallucination spans over six generators | [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth) |

<br>

## Layout

```
reproduce.sh          one target per table, figure and appendix section   ← start here
run_all.sh            the whole pipeline in dependency order
paper_figures.sh      copies the paper's figures into a LaTeX folder
scriptmetric/         the library: metric.py (SCRIPT, profile, distance, identify), betting.py (SCRIPT-Seq)
pipeline/             the paper: behaviour/  metric/  external/  figures/  figures/handmade/  paper_tables.py
assets/               images used by this README
docs/                 explainer page
out/  raw/            outputs and downloads, created on first run, git-ignored
```

<br>

## Citation and licence

Code is released under the MIT licence (`LICENSE`). Each dataset keeps its own licence, listed there; the Cognitive Atrophy Benchmark is **CC BY-NC 4.0** (non-commercial).

```bibtex
@misc{script2027,
  title  = {SCRIPT: Measuring Behavioural Scriptedness from Span Annotations},
  author = {Anonymous},
  year   = {2027},
  note   = {Under review}
}
```
