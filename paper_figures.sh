#!/usr/bin/env bash
# Copy the figure files the paper includes into a LaTeX figures folder.
#   bash paper_figures.sh ../figures
# Run after run_all.sh (or after the individual figure scripts); the two
# hand-built figures are exported from pipeline/figures/handmade/ and copied as well.
set -euo pipefail
cd "$(dirname "$0")"
DEST="${1:-../figures}"
mkdir -p "$DEST"
for f in fig_anatomy fig_behaviour_language fig_behaviour_overview fig_betting fig_case_study \
         fig_evidence_ladder fig_method_faithful fig_momentum fig_multiturn fig_null_validation \
         fig_results_closing fig_results_main fig_transition_lift fig_worked_example; do
  cp "out/figures/$f.png" "$DEST/"
done
cp pipeline/figures/handmade/method_figure/script_method_figure.png "$DEST/"
cp pipeline/figures/handmade/profile_figure/script_profile_figure.png "$DEST/"
echo "copied 16 figures to $DEST"
