#!/usr/bin/env bash
# Reproduce every table and figure from an empty checkout.
#
#   bash run_all.sh
#
# Inputs are downloaded by pipeline.data.get_data into raw/; outputs land in out/
# (tables/, figures/, derived/, checkpoints/). Bootstrap stages are chunked and
# checkpointed, so they can run in parallel shells or resume after an interruption.
# The LLM-annotator experiment needs an API key and is run separately (see README).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3} -m"

echo "=== data ==================================================="
$PY pipeline.data.get_data

echo "=== benchmark behaviour analysis ==========================="
$PY pipeline.behaviour.attributes
$PY pipeline.behaviour.spans
$PY pipeline.behaviour.therapist
$PY pipeline.behaviour.dynamics
$PY pipeline.behaviour.phrases

echo "=== the span layer and the metric =========================="
$PY pipeline.metric.events
$PY pipeline.metric.validate
$PY pipeline.metric.identification
$PY pipeline.metric.identification_baselines
$PY pipeline.metric.profiles

echo "=== robustness ============================================="
for c in 0:100 100:200 200:300 300:400; do $PY pipeline.metric.robustness "BOOT:$c"; done
$PY pipeline.metric.robustness COLLECT
$PY pipeline.metric.robustness ALL
for q in Q1 Q2 Q3 Q4; do $PY pipeline.metric.sensitivity "$q"; done
for r in R1 R2 R3 R4 R5 R6; do $PY pipeline.metric.sensitivity "Q5A:$r"; done
$PY pipeline.metric.sensitivity Q5COLLECT
for s in NULLS LEN SHUF CARD; do $PY pipeline.metric.nulls_ablations "$s"; done
for r in R1 R2 R3 R4 R5 R6; do $PY pipeline.metric.annotator_tiers "$r"; done
$PY pipeline.metric.annotator_tiers COLLECT
$PY pipeline.metric.jury
$PY pipeline.metric.tie_robustness

echo "=== extensions ============================================="
$PY pipeline.metric.multiturn
$PY pipeline.metric.quality_matched
$PY pipeline.metric.conformity
$PY pipeline.metric.conformity_length_control
$PY pipeline.metric.therapist_baseline

echo "=== external corpora ======================================="
$PY pipeline.external.anchors
$PY pipeline.metric.ceiling_extrapolation
$PY pipeline.external.empathy_tactics
$PY pipeline.external.empathy_checks
$PY pipeline.external.mint_surface_layer
if [ -f out/derived/llm_span_events.csv ]; then $PY pipeline.external.listening; $PY pipeline.external.llm_agreement; $PY pipeline.metric.therapist_paired ALL; $PY pipeline.external.listening_extra; fi

echo "=== sequential testing ====================================="
$PY pipeline.metric.betting_validity V1 --streams=200 --naive=60
$PY pipeline.metric.betting_validity V2
$PY pipeline.metric.betting_validity V3
$PY pipeline.metric.betting_applications ALL

echo "=== sensitivity of the estimator and the extensions ========"
$PY pipeline.metric.sensitivity_extra ALL

echo "=== figures ================================================"
$PY pipeline.metric.figure_data
for f in behaviour_overview behaviour_levels explanatory anatomy case_study worked_example \
         method_faithful momentum multiturn null_validation betting; do
  $PY "pipeline.figures.$f"
done
# Figure 2 (hand-built) needs the LLM-annotator layer
if [ -f out/tables/listening_budget.csv ]; then
  ( cd pipeline/figures/handmade/main_figures && "${PYTHON:-python3}" build_listening_figure.py )
fi
$PY pipeline.figures.results_closing

echo "=== paper tables ==========================================="
$PY pipeline.paper_tables

echo
echo "done. tables -> out/tables/   figures -> out/figures/"
