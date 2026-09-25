#!/usr/bin/env bash
# Copy the figure files the paper includes into a LaTeX figures folder.
#   bash paper_figures.sh ../figures
# Run after run_all.sh (or after the individual figure scripts); every figure,
# including the three hand-built ones, is written to out/figures/.
set -euo pipefail
cd "$(dirname "$0")"
DEST="${1:-../figures}"
mkdir -p "$DEST"
for f in script_method_figure script_listening_figure fig_results_closing \
         fig_anatomy fig_behaviour_language fig_behaviour_overview fig_betting fig_case_study \
         fig_evidence_ladder fig_method_faithful fig_momentum fig_multiturn fig_null_validation \
         fig_transition_lift fig_worked_example script_profile_figure; do
  cp "out/figures/$f.png" "$DEST/"
done
echo "copied 16 figures to $DEST"
