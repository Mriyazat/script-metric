#!/usr/bin/env bash
# Everything, from an empty checkout to every table and figure in the paper.
#
#   bash run_all.sh
#
# Inputs are downloaded from HuggingFace and GitHub by stage 00; outputs land
# in out/ (tables/, figures/, derived/, checkpoints/). Nothing is committed.
#
# The bootstrap stages are chunked so they can run in parallel or resume after
# an interruption: each replicate has its own fixed seed and is checkpointed
# under out/checkpoints/, so re-running a chunk is a no-op.
set -euo pipefail
cd "$(dirname "$0")/pipeline"
PY=${PYTHON:-python3}

echo "=== 0. data ==============================================="
$PY 00_get_data.py

echo "=== 1. the span layer ====================================="
$PY 01_extract_events.py
$PY 02_validate_metric.py

echo "=== 2. profiles and identity =============================="
$PY 14_identification_curves.py      # 03 reads its curves
$PY 03_profile_suite.py

echo "=== 3. robustness ========================================="
# 400 cluster-bootstrap replicates, as in the paper. Chunks are independent
# and checkpointed, so they can also be launched in parallel shells.
$PY 04_robustness_suite.py BOOT:0:100
$PY 04_robustness_suite.py BOOT:100:200
$PY 04_robustness_suite.py BOOT:200:300
$PY 04_robustness_suite.py BOOT:300:400
$PY 04_robustness_suite.py COLLECT
$PY 04_robustness_suite.py ALL

echo "=== 4. sensitivity ========================================"
for q in Q1 Q2 Q3 Q4; do $PY 05_sensitivity_suite.py "$q"; done
for r in R1 R2 R3 R4 R5 R6; do $PY 05_sensitivity_suite.py "Q5A:$r"; done
$PY 05_sensitivity_suite.py Q5COLLECT

echo "=== 5. nulls and ablations ================================"
for s in NULLS LEN SHUF CARD; do $PY 06_null_ablation_suite.py "$s"; done

echo "=== 6. annotator checks ==================================="
for r in R1 R2 R3 R4 R5 R6; do $PY 07_loao_tier_check.py "$r"; done
$PY 07_loao_tier_check.py COLLECT
$PY 10_jury_check.py

echo "=== 7. contribution 3: anytime-valid testing by betting ==="
$PY 15_betting_validity.py V1 --streams=200 --naive=60
$PY 15_betting_validity.py V2
$PY 15_betting_validity.py V3
$PY 16_betting_applications.py ALL

echo "=== 8. extensions and anchors ============================="
$PY 08_multiturn_extension.py
$PY 09_quality_matched_check.py
$PY 11_external_anchors.py
$PY 12_therapist_baseline.py
$PY 17_ceiling_extrapolation.py

echo "=== 9. figures ============================================"
$PY 13_figure_data.py
for f in 20_fig_script_metric 21_fig_explanatory 22_fig_anatomy \
         23_fig_five_scripts 24_fig_results_hero 25_fig_case_study \
         26_fig_worked_example 27_fig_hero 28_fig_method_faithful \
         29_fig_momentum 30_fig_multiturn 31_fig_null_validation \
         32_fig_betting; do
  $PY "$f.py"
done

echo
echo "done. tables -> out/tables/   figures -> out/figures/"
