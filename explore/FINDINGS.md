# Findings beyond the paper

Nothing in the paper itself was changed. Sections 1 and 2 are exploratory
probes; **section 3 is a full third contribution**, now built into the
repository proper.

```
explore/e1_user_coupling.py               does the script ignore the user?
explore/e1b_prompt_vs_model_controlled.py the controlled version of that
explore/e2_annomi.py                      SCRIPT on AnnoMI human counselling

script_betting.py                         Contribution 3 — the method
pipeline/15_betting_validity.py           validity, power, sensitivity
pipeline/16_betting_applications.py       calibration, ceiling, A/B, transfer
pipeline/32_fig_betting.py                the figure
explore/e3_sequential_betting.py          the original prototype (superseded)
```

---

## 0. Where the two papers sit relative to this work

### Orabona & Tommasi, *Training Deep Networks without Learning Rates Through Coin Betting* (NeurIPS 2017)

Not directly applicable — SCRIPT is an estimator, not an optimisation problem,
and there is no learning rate anywhere in it. Two things do carry over.

**The framing is exact, not metaphorical.** Mutual information *is* the
expected log-wealth of an optimal (Kelly) gambler. `I(L;X)/H(L)` is literally
the fraction of a no-information gambler's log-wealth that a gambler who knows
the position earns. The `guessing_game` panel of the explainer already tells
the story informally; the coin-betting reduction is what makes it a definition
rather than an analogy, and it is the bridge to the second paper.

**A framing SCRIPT could borrow — not a defect it has.** SCRIPT has two knobs:
bin count `B = 10` and shuffle count `S = 200`. Neither is a weakness. `S` is a
Monte-Carlo budget, not a modelling choice — more shuffles only sharpen the
same estimate, and `shuffle_count_stability.csv` shows the score is flat in it
to the fourth decimal. `B` is a genuine modelling choice, and it is already
handled the way the field expects, with a reported sweep (`bin_sweep.csv`).
The coin-betting literature's ambition is stronger — *remove* the knob rather
than sweep it — and section 3 happens to remove `S` for free. That is an
opportunity, not a hole a reviewer will find.

### Chen & Wang, *Online Detection of LLM-Generated Texts via Sequential Hypothesis Testing by Betting* (arXiv:2410.22318)

Directly transferable, and it closes a real hole. Implemented in section 3.

---

## 1. Does the script ignore the user's message?

**Answered in the paper: no.** Contribution 2 never touches the user's turn.
The metric reads `(label, position)` and nothing else, and there is no term
anywhere that relates behaviour to input. This is the most obvious reviewer
question and the paper currently has no answer to it: if the user's message
explained arrangement equally well, "decided before reading the input" would
be unsupported.

The corpora already contain what is needed. Every item carries clinician
ratings **of the user's turn** — `user_request_info`, `user_evocative`,
`user_sensitivity`, `user_underlying`, `user_typicality`, plus topic — at 100%
coverage. These instantiate almost exactly the controlled contrasts you
proposed (wants to be heard / asks for advice / uncertain / sensitive /
unstated belief / topic), without needing to generate anything.

### 1a. Responsiveness, on SCRIPT's own scale

Define the mirror image of choreography:

```
A_u = I(L ; u | X) / H(L)
```

null-calibrated by permuting the user-state `u` across responses within corpus
— which breaks the user↔behaviour link while holding every label sequence,
position and marginal fixed, exactly as SCRIPT's null breaks the
label↔position link.

Because a 3-level user variable and a 10-bin position variable do not have the
same capacity, position is re-binned to the *same number of levels* as each
user variable. Then the two numbers are strictly like-for-like.

| user property | A_u | z | position at matched levels | ratio |
|---|---:|---:|---:|---:|
| explicitly asks for information/advice | 0.0040 | 21.6 | 0.0224 (2 lv) | **5.6×** |
| emotionally evocative | 0.0037 | 14.7 | 0.0344 (3 lv) | **9.3×** |
| sensitive / risk-bearing | 0.0088 | 39.8 | 0.0224 (2 lv) | **2.5×** |
| underlying unstated issue | 0.0055 | 26.6 | 0.0344 (3 lv) | **6.3×** |
| typicality of presentation | 0.0135 | 49.8 | 0.0344 (3 lv) | **2.5×** |
| topic (22 levels) | 0.0380 | 61.9 | 0.0541 (22 lv) | **1.4×** |

**Reading.** The user's state is *not* ignored — every effect is real and
highly significant. But at matched capacity, where a behaviour sits in the
reply explains 1.4× to 9.3× more of it than any clinician-rated property of
what the user actually said. The sharpest single number: whether the user
**explicitly asked for advice** — the property most likely to require a
different reply — moves the arrangement 5.6× less than position does.

This is a stronger and more defensible claim than "the script ignores the
user", and it is the claim the data supports.

### 1b. Matched head-to-head: same prompt vs same model

Every prompt is answered by all five models, so for any held-out reply there
are two reference sets of exactly four other replies: four replies to the
**same prompt by other models**, or four replies by the **same model to other
prompts**. Predict the held-out reply's arrangement from each (mean per-event
log-likelihood). Matched by construction — four replies either side.

| | same-model reference wins | mean margin (nats/event) |
|---|---:|---:|
| raw (composition + arrangement) | 14.7% | −0.789 (t = −20.0) |
| **arrangement only** (within-reply shuffle removes composition) | 23.4% | −0.249 (t = −13.1) |

The same-prompt reference wins decisively. Two thirds of that advantage is
*composition* — which behaviours appear, which Contribution 1 already
attributes to the prompt — but a real arrangement advantage survives the
shuffle control.

### 1c. The controlled version (`e1b_prompt_vs_model_controlled.py`)

Two confounds made 1b uninterpretable: the two references are not equally good
*as profiles* (the same-prompt set pools four models and is smoother), and
replies to one prompt share a length, hence bin coverage. Both are fixed by
adding a third reference — four replies by **other models to other prompts** —
and measuring each effect as a lift over that common baseline, with all three
reference sets matched on per-reply event counts.

| lift over the common baseline (nats/event, arrangement only) | lift | t | positive in |
|---|---:|---:|---:|
| same PROMPT (user's message) | **+0.160** | 7.2 | 72% |
| same MODEL | **+0.043** | 2.2 | 56% |
| difference (prompt − model) | +0.117 | 4.9 | — |

The confounds were not the explanation: **at the level of a single reply, the
user's message carries about 3.7× more arrangement information than the model
does.** Both are real; the prompt effect is the bigger one.

**Why this does not break the paper — and what to change.** Three things.

1. It is a *different quantity*. This is reply-to-reply similarity at n = 4.
   SCRIPT is a corpus-scale regularity pooled over ~800 replies, and the
   identification result is built on pooled profiles, where the model *is*
   recoverable (0.56 at k=5, 0.73 at k=20). No published number is contradicted.
2. **The paper already contains the control that answers this.** The human
   therapist answered *identical prompts*. Whatever arrangement the user's
   message imposes, it is imposed on the therapist too — and the therapist
   still scores lowest of the six speakers. Any "the arrangement just comes
   from the content" objection is already handled by that comparison, which
   should be promoted from a supporting result to the primary defence.
3. What does need softening is rhetoric, not results: phrases implying the
   behaviour is fixed *before* the input is read overshoot. The supportable
   claim is 1a's — at matched capacity, position explains 1.4–9.3× more than
   any rated property of the user's turn — plus the therapist anchor.

The clean way to say it: **the user's message shapes a reply's arrangement, and
the models still impose more fixed structure on top of it than a human
professional does on the same messages.**

---

## 2. Another annotated dataset — AnnoMI

**Answered in the paper: partly.** Three external corpora are already in
(RAGTruth, WMT24, data-to-text) but all three are *error-span* schemes on
non-conversational tasks. None is a conversation, a clinical coding system, or
a human speaker. AnnoMI is all three, and it is the single best external check
available.

**Unit.** AnnoMI has no span layer, so a session is the unit and an
utterance's position is its place in the session timeline. This measures
*session* choreography, not within-reply choreography. Every comparison below
is against the benchmark re-scored at the same unit and the same alphabet —
never against the paper's headline 0.10.

133 sessions, 4,882 therapist utterances, 4 MI behaviour codes
(question / reflection / therapist_input / other), expert high-vs-low quality
label.

### Q1 — Is human motivational interviewing scripted?

**Yes, and in a completely different way from LLMs.**

| | SCRIPT | z | C (choreography) | M (momentum) |
|---|---:|---:|---:|---:|
| all human MI | 0.0471 | 24.5 | 0.0111 | **0.0360** |
| high-quality MI | 0.0432 | 21.3 | 0.0097 | **0.0335** |
| low-quality MI | 0.0546 | 4.0 | **0.0485** | 0.0061 |

Human MI is **momentum-driven**: counsellors chain moves (reflection follows
question follows reflection) but do not park behaviours at fixed points in the
hour. LLMs within a reply are the mirror image — choreography-dominant.

### Q2 — Are LLMs more scripted than human counsellors?

At the session unit, on the MI alphabet: **no, the opposite.**

| | SCRIPT | z | C | M |
|---|---:|---:|---:|---:|
| human MI (high quality) | **0.043** | 21.3 | 0.010 | 0.034 |
| Qwen | 0.025 | 17.0 | 0.002 | 0.023 |
| Llama | 0.016 | 7.0 | 0.001 | 0.015 |
| GPT | 0.019 | 11.6 | 0.003 | 0.016 |
| Claude | 0.029 | 15.6 | 0.006 | 0.023 |
| Gemini | 0.026 | 13.3 | 0.002 | 0.024 |

This is **not** a contradiction of the paper — it is independent confirmation
of its multi-turn result. The paper already shows boundary momentum is null
for every model: the LLM script restarts each turn. Measured across a whole
session, there is therefore little for SCRIPT to find in an LLM, while a human
counsellor's session-level discipline shows up clearly. Independent data, an
independent coding scheme, and human speakers reproduce the paper's own
"the script restarts every turn" finding from the other direction.

### Q3 — Is low-quality MI more scripted than high-quality MI?

**Not in level — but decisively in kind.** Low-quality MI has 10× less data,
and SCRIPT is biased down at small n, so the raw comparison is confounded.
Subsampling high-quality to the same 23 sessions (40 draws):

| | matched high-quality | low-quality | gap | permutation p |
|---|---:|---:|---:|---:|
| SCRIPT | 0.0467 ± 0.0134 | 0.0546 | +0.0079 (0.6 SD) | 0.27 |
| **C** | 0.0128 ± 0.0070 | 0.0485 | **+0.0357 (5.1 SD)** | 0.024 |
| **M** | 0.0338 ± 0.0105 | 0.0061 | **−0.0277 (2.6 SD)** | 0.024 |

**Good and bad counselling are equally scripted, and scripted in opposite
ways.** Good MI chains its moves responsively; bad MI parks them at fixed
points in the session. The *level* cannot tell them apart; the C/M
decomposition can.

This is the first external, expert-rated evidence that SCRIPT's decomposition
carries clinical meaning, and it directly justifies the paper's choice to
split the score rather than report one number. Caveats: 23 low-quality
sessions, and permutation p is floored at 1/41 by the number of subsamples.

### Q4 — Does SCRIPT behave sensibly under an established coding system?

Yes. Random-label control on AnnoMI: SCRIPT = **0.0004**, z = 0.3 — the null
lands on zero under a coding scheme the metric has never seen.

---

## 3. Contribution 3 — SCRIPT-Seq, anytime-valid scriptedness testing

**Now implemented as a first-class part of the repository**, not an
exploration script:

```
script_betting.py                 the method — standalone, numpy + pandas, CLI
pipeline/15_betting_validity.py   V1 validity, V2 power, V3 sensitivity
pipeline/16_betting_applications.py  A1 calibration, A2 ceiling audit,
                                     A3 A/B regression test, A4 transfer
pipeline/32_fig_betting.py        the figure
```

### 3.0 The gap it fills

SCRIPT answers *given this corpus, is the structure real?* Three of the four
uses the paper proposes — a deployment ceiling, a regression test across a
release, a live monitor — are streaming decisions, and a fixed-sample `z`
cannot make them. **Measured: recomputing `z` every 20 replies and stopping the
first time it clears 1.96 gives a false-positive rate of 27%.**

### 3.1 The construction

```
at round t:  profile built from replies 1..t-1 only          (past-measurable)
             e_t = loglik(reply t) − mean over K=20 within-reply shuffles
             W_t = W_{t-1}(1 + θ_t e_t),  θ_t ≥ 0 by Online Newton Step
             reject when W_t ≥ 1/α
```

`E[e_t | F_{t-1}] = 0` **exactly** under H0, because SCRIPT's null already is an
exchangeability statement: the real reply is itself distributed as one of its
own shuffles. Nothing is assumed about label distributions, reply lengths,
annotators or sample size. Ville's inequality then gives level-α validity
*simultaneously at every t*.

Four tests ship: existence (`eps=0`), ceiling (`eps>0`, composite), paired
two-sample comparison, and an anytime-valid lower confidence sequence.

### 3.2 V1 — validity (`tables/betting_validity.csv`)

200 independent H0 streams (labels permuted within each reply), α = 0.05,
watched to 400 replies:

| method | false positives |
|---|---:|
| **betting (Ville)** | **0 / 200 = 0.000** — guarantee ≤ 0.05 |
| naive: recompute `z`, peek every 20 replies | **17 / 62 = 0.274** |

### 3.3 V2 — power (`tables/betting_power.csv`)

| scope | detected | median replies to declare |
|---|---:|---:|
| Qwen | 30/30 | 41 (IQR 39–47) |
| Llama | 30/30 | 33 (IQR 29–37) |
| GPT | 30/30 | 36 (IQR 34–39) |
| Claude | 30/30 | 37 (IQR 32–41) |
| Gemini | 30/30 | 37 (IQR 32–46) |

**Two results here matter more than the headline.**

*It rescues the small corpora.* The paper says of counselchat and pair: "the
small single-turn corpora [≈600 events] have low z … judge them on the pooled
column." The sequential test declares them templated anyway — counselchat
29/30 at a median of 33 replies, pair 30/30 at 26 — on corpora holding only
48 and 50 replies. A claim the published analysis had to withhold becomes
available.

*It declines exactly where the paper says it should.* Per annotator: R1–R4 and
R6 all detected (median 20–95 replies); **R5 detected 0/10** — and R5 is
precisely the jury the paper's own robustness work excludes as too sparse to
support sequence claims. The test refuses to declare on the one jury that
cannot support the declaration.

### 3.4 V3 — sensitivity (`tables/betting_scaling.csv`)

The real systems all have similar edges, so they cannot trace a power curve.
Diluting — shuffling a fraction `p` of replies — interpolates continuously
between a fully templated system and the null:

| template kept | edge | detected | median τ |
|---:|---:|---:|---:|
| 100% | +0.619 | 8/8 | 44 |
| 75% | +0.347 | 8/8 | 63 |
| 50% | +0.174 | 8/8 | 102 |
| 25% | +0.058 | 5/8 | 278 |
| 10% | +0.014 | 0/8 | — |
| 0% (null) | +0.001 | 0/8 | — |

**Weakest template caught in every stream: 50% of the real signal.** The
log τ vs log edge slope is −0.66 rather than the −2 of Chen & Wang's bound,
because at these effect sizes the stopping time is burn-in-limited (the profile
must accumulate before any edge exists) rather than signal-limited.

### 3.5 A1 — calibration (`tables/betting_calibration.csv`)

The edge is in nats/event, not on the SCRIPT scale, so a ceiling stated as
"SCRIPT ≤ 0.05" has to be converted. Across the dilution sweep plus the five
real models:

```
SCRIPT = 0.1593 × edge + 0.0061          r = 0.983,  n = 19
```

| ceiling | eps |
|---|---:|
| SCRIPT = 0.02 | 0.087 nats/event |
| SCRIPT = 0.05 | 0.275 |
| SCRIPT = 0.08 | 0.464 |

### 3.6 A2 — the deployment audit (`tables/betting_threshold.csv`)

Composite null H0: δ ≤ eps. Rejecting means the system sits *above* the
ceiling, not merely above zero.

| ceiling | Qwen | Llama | GPT | Claude | Gemini |
|---|---|---|---|---|---|
| SCRIPT 0.02 | 6/6 @ 68 | 6/6 @ 40 | 6/6 @ 46 | 6/6 @ 54 | 6/6 @ 45 |
| SCRIPT 0.05 | 6/6 @ 156 | 6/6 @ 75 | 6/6 @ 105 | 6/6 @ 91 | 6/6 @ 97 |
| SCRIPT 0.08 | **4/6** @ 419 | 6/6 @ 149 | 6/6 @ 235 | 6/6 @ 241 | 6/6 @ 311 |

Behaves exactly as a threshold test should: the closer the ceiling sits to a
system's true value, the more evidence it takes, and Qwen — the least scripted
of the five — is the one that stops clearing 0.08 reliably.

### 3.7 A3 — the regression test (`tables/betting_ab.csv`)

Paired sequential comparison, 5/20 ordered pairs decided within 500 replies:

```
Claude > Gemini   t = 164        Llama  > Qwen     t = 225
Llama  > Gemini   t = 177        Claude > Qwen     t = 230
                                 GPT    > Qwen     t = 355
```

Broadly the paper's two-tier structure {Llama, Claude} > {Qwen, GPT, Gemini},
reached without a cluster bootstrap or Holm correction, and correctly leaving
Claude/Llama undecided.

**Honest caveat.** The edge and SCRIPT do *not* rank the five models
identically (`tables/betting_edge_vs_script.csv`):

| | SCRIPT | rank | edge | rank |
|---|---:|---:|---:|---:|
| Claude | 0.1104 | 1 | 0.606 | 2 |
| Llama | 0.1047 | 2 | 0.703 | 1 |
| Gemini | 0.1027 | 3 | 0.577 | 4 |
| Qwen | 0.0983 | 4 | 0.499 | 5 |
| GPT | 0.0978 | 5 | 0.602 | 3 |

The two agree strongly across systems with genuinely different amounts of
structure (r = 0.98 in the dilution sweep) but disagree inside the band SCRIPT
itself calls a tie — which is why `GPT > Qwen` appears above for a pair whose
SCRIPT values differ by 0.0005. **The A/B test should be used to detect
substantial differences, not to rank near-tied systems.** That is the same
caution the paper already states for the score itself.

### 3.8 A4 — transfer (`tables/betting_transfer.csv`)

The same test, unchanged, on corpora from other tasks. A test that fires
everywhere would be worthless:

| corpus | SCRIPT | z | edge | rejects? | τ |
|---|---:|---:|---:|---|---:|
| counselling (Claude, 20 codes) | 0.110 | 56.2 | 0.663 | **yes** | 44 |
| AnnoMI human counsellors | 0.047 | 24.5 | 0.098 | **yes** | 56 |
| data-to-text error spans | 0.049 | 11.6 | 0.052 | **yes** | 320 |
| RAGTruth hallucination | 0.014 | 6.7 | 0.038 | no | — |
| WMT24 MT error spans | 0.026 | 3.0 | 0.001 | no | — |

The sequential test inherits the metric's discrimination and its ordering:
fast on counselling and AnnoMI, slow on data-to-text, silent on the two
anchors the published ruler already calls content-driven.

### 3.9 Byproduct — an anytime-valid lower bound

Inverting the ceiling test over a grid of eps gives a running lower bound on
the effect that is valid at every t simultaneously, so it can be watched
continuously and stopped on without any correction:

| replies | Qwen | Llama | GPT | Claude | Gemini |
|---:|---:|---:|---:|---:|---:|
| 50 | −0.03 | 0.18 | 0.20 | 0.08 | 0.02 |
| 100 | 0.18 | 0.33 | 0.35 | 0.38 | 0.17 |
| 300 | 0.38 | 0.58 | 0.52 | 0.58 | 0.45 |

A fixed-sample confidence interval cannot be read this way.

### 3.10 What is still missing

- The edge is a *proxy* for SCRIPT (r = 0.98 across systems, but not
  rank-faithful within a tie). A payoff whose mean is SCRIPT itself would be
  better and does not obviously exist — SCRIPT is a corpus-level functional
  and does not decompose into a per-reply average.
- Validity is empirical here (0/200) plus the theoretical guarantee inherited
  from Ville; the exchangeability argument is stated but not written up as a
  proof.
- **One residual gap in the guarantee.** The bettor clips `theta` using a
  bound derived from the edges seen *so far*, so nothing formally rules out
  `1 + theta_t e_t < 0` on the next reply, which would break the non-negativity
  Ville's inequality needs. `Bettor.observe` guards with `max(·, 0)`, and that
  guard is what would technically invalidate the bound if it ever fired.
  Instrumented over 24,000 wealth updates across all five models and 40 null
  streams: **the factor never went negative, and its smallest observed value
  was 0.349.** So the guard is inert on this data — but a paper version should
  either adopt Chen & Wang's explicit hint `d_{t+1} >= |e_{t+1}|` or state a
  bound on the edge and clip to it.
- `B = 10` bins is still a hyperparameter. `S`, the permutation budget,
  disappears entirely in this formulation.
- Detection times assume replies arrive in random order. A monitor seeing
  correlated batches (one annotator, one topic) would need the ordering
  argument checked.

### 3.11 Verdict on Contribution 3

It stands on its own. It fixes a genuine invalidity in how the metric would be
used in practice, it is 20× more data-efficient for the declaration the paper
actually wants to make, it recovers claims the fixed-sample analysis had to
withhold on small corpora, it declines on exactly the jury and the corpora
where declining is correct, and it adds a ceiling audit and a regression test
the published statistic cannot support at all. The main limitation is honest
and stateable: it tests a monotone proxy for SCRIPT rather than SCRIPT itself.

---

## Appendix — earlier prototype notes

## 3. Anytime-valid SCRIPT by betting

**The gap.** SCRIPT as published is a fixed-sample statistic. The paper
proposes it as a deployment threshold and a regression test — uses where
evidence arrives over time and you want to act as soon as it is sufficient.
If you recompute `z` as annotations accumulate and stop the first time it
crosses 1.96, you have no error control at all.

**The port.** Chen & Wang's construction needs a payoff that is exactly
mean-zero under the null. SCRIPT's null already *is* an exchangeability
statement, which makes the fit unusually clean:

```
at round t:  build the profile from replies 1..t-1     (F_{t-1}-measurable)
             a_t = per-event log-lik of reply t under it
             b_t = same, averaged over 20 within-reply label shuffles
             g_t = b_t - a_t              E[g_t | F_{t-1}] = 0 under H0
             wealth  W_t = W_{t-1}(1 - g_t θ_t),  θ_t by Online Newton Step
             reject when W_t ≥ 1/α
```

No reference corpus, no held-out split, no threshold to tune, no permutation
budget, and no need to fix the sample size in advance. Ville's inequality
makes it level-α *simultaneously at every t*.

### Validity — 200 independent H0 streams (α = 0.05)

| method | false-positive rate |
|---|---:|
| **betting (Ville)** | **0/200 = 0.000** (guaranteed ≤ 0.05) |
| naive: recompute z, peek every 10 replies to t = 300 | **14/53 = 0.26** |

Peeking at the paper's own statistic gives a 26% false-positive rate. The
betting test is valid, and conservative as supermartingale bounds always are.

### Power — replies needed to certify a real model

| model | detected | median τ | 90th pct |
|---|---:|---:|---:|
| Qwen | 30/30 | 46 | 57 |
| Llama | 30/30 | 33 | 39 |
| GPT | 30/30 | 41 | 57 |
| Claude | 30/30 | 38 | 47 |
| Gemini | 30/30 | 39 | 59 |

**About 40 annotated replies** is enough to declare a frontier model templated
with a controlled false-positive rate — against ~800 replies per model for the
published fixed-sample analysis. Wealth then keeps compounding: 10¹⁰–10¹⁵ by
t = 400, while the H0 stream sits at 2.2 and never crosses.

See `out/explore/fig_betting_wealth.png`.

### Why this is worth adding

It converts SCRIPT from a corpus statistic into a **monitoring instrument with
a stopping rule** — which is what the paper's "deployment threshold",
"regression test" and "provenance audit" applications actually require. It
removes one of the two hyperparameters. And it answers "how much annotation do
I need?" with a number instead of a shrug.

---

## Summary

| question | status | outcome |
|---|---|---|
| user–response coupling measured? | **not in paper** | now measured; position beats every user variable 1.4–9.3× at matched capacity |
| tested on other annotated data? | **partly** | 3 error-span corpora were in; AnnoMI adds conversation + clinical codes + humans |
| does structure change when the required behaviour changes? | **not in paper** | tested observationally via clinician user-state ratings; effects real but small |
| AnnoMI | **new** | human MI is momentum-scripted; bad MI is choreography-scripted; null control clean |
| anytime-valid inference | **now Contribution 3** | fully implemented in `script_betting.py` + stages 15/16/32; 0/200 false positives vs 27% for naive peeking; 33–41 replies to certify |
| prompt-coupled arrangement | **new, qualifying** | real (3.7× the model effect at reply level); already answered by the therapist-on-identical-prompts control |

### Verdict on the contribution

Nothing here falsifies SCRIPT. The estimator does what it claims: it reads zero
on known-zero data, it separates a pure-slot from a pure-chain generator, it
reproduces from source, and it now behaves correctly on a clinical coding
system it has never seen. Two findings make it stronger — AnnoMI shows the C/M
split tracks expert-rated counselling quality where the level cannot, and the
betting version turns it into a monitoring instrument. One finding narrows a
rhetorical claim rather than a measured one.

Three things to do, in order of value:

1. **Add AnnoMI.** Highest return per unit of work. It answers "does this
   generalise beyond your own benchmark and your own coding manual" with an
   independent clinical scheme, human speakers, and an expert quality label —
   and the C/M result is a new claim, not a robustness check.
2. **Add the responsiveness term (1a).** Closes the single most predictable
   reviewer question with a table you can compute in minutes.
3. **Soften the "decided before reading the input" phrasing** and promote the
   therapist-on-identical-prompts comparison to the primary defence.

4. **Contribution 3 (SCRIPT-Seq).** Now built and measured end to end —
   see section 3 above. It is genuinely separable from Contribution 2 and
   could carry its own paper, but as a third contribution it does real work:
   it makes three of the four proposed applications statistically legitimate.
