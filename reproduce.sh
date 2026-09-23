#!/usr/bin/env bash
# One command per table and figure of the paper.
#
#   bash reproduce.sh data              download and verify the inputs (once)
#   bash reproduce.sh table2            one main-text item ...
#   bash reproduce.sh main              ... or all of Section 5 (Tables 2-4, Figures 2-3)
#   bash reproduce.sh appendix          every appendix table and figure
#   bash reproduce.sh B2                one appendix section (A .. D, B1 .. B5, C1 .. C6)
#   bash reproduce.sh list              show the targets
#
# Outputs go to out/tables/ and out/figures/. Targets that need the blind LLM
# re-annotation (Table 2's LLM block, Table 3, Figure 2, B1, B2) read the shipped
# event file data/llm_span_events.csv and need no API key. Only the span-by-span
# agreement table of Appendix A needs the per-reply cache (an API key regenerates it).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3} -m"

run() { echo; echo "▶ $*"; $PY "$@"; }
need_events() { [ -f out/derived/span_events.csv ] || run pipeline.metric.events; }
need_llm() {
  # the blind LLM layer ships as an event file (labels + positions, no text): data/llm_span_events.csv
  [ -f out/tables/llm_annotator_comparison.csv ] || run pipeline.external.llm_layer_from_events
}
need_listening() { need_llm; [ -f out/tables/listening_budget.csv ] || run pipeline.external.listening; }
need_anchors()   { [ -f out/tables/anchor_wmt24.csv ] || run pipeline.external.anchors; }
need_testbed2()  { [ -f out/tables/mint_systems.csv ] || run pipeline.external.empathy_checks
                   [ -f out/tables/mint_surface_layer.csv ] || run pipeline.external.mint_surface_layer; }
need_robust()    { [ -f out/tables/pairwise_tests.csv ] || {
                     for c in 0:100 100:200 200:300 300:400; do run pipeline.metric.robustness "BOOT:$c"; done
                     run pipeline.metric.robustness COLLECT; run pipeline.metric.robustness ALL; }; }

target() {
  case "$1" in
    data)     run pipeline.data.get_data ;;

    # ---------------------------------------------------------------- main text
    table2)   need_events; run pipeline.metric.validate; need_llm; need_robust
              run pipeline.metric.therapist_paired PAIRED
              echo "→ out/derived/validation_results.csv  out/tables/llm_annotator_comparison.csv"
              echo "→ out/tables/therapist_paired_C.csv  out/tables/pairwise_tests.csv  out/verify_report.json (V3)" ;;
    table3)   need_events; need_listening
              echo "→ out/tables/listening_arc.csv  listening_user_state.csv  listening_budget.csv" ;;
    table4)   need_events; need_testbed2; run pipeline.metric.conformity; run pipeline.metric.conformity_length_control
              echo "→ out/tables/conformity_tb2_pairs_own.csv  conformity_tb1_pairs_own.csv  conformity_length_control.csv"
              echo "→ out/tables/mint_systems.csv  mint_surface_layer.csv" ;;
    fig1)     ( cd pipeline/figures/handmade/method_figure && python3 build_figure.py && python3 export_figure.py )
              echo "→ pipeline/figures/handmade/method_figure/script_method_figure.png" ;;
    fig2)     need_events; need_listening; run pipeline.figures.results_main
              echo "→ out/figures/fig_results_main.png" ;;
    fig3)     need_events; run pipeline.metric.identification; run pipeline.metric.identification_baselines
              need_anchors; run pipeline.figures.results_closing
              echo "→ out/figures/fig_results_closing.png  out/tables/identification_curves.csv  identification_baselines.csv" ;;
    numbers)  run pipeline.paper_tables; echo "→ out/tables/latex/numbers.json  (every number quoted in the prose)" ;;
    main)     for t in table2 table3 table4 fig2 fig3 numbers; do target $t; done ;;

    # ---------------------------------------------------------------- appendix
    A)        need_llm; need_anchors
              if [ -d out/derived/llm_annotator ]; then run pipeline.external.llm_agreement;
              else echo "(agreement table skipped: needs the per-reply annotator cache; see README)"; fi
              echo "→ out/tables/llm_agreement*.csv  external_anchors.csv" ;;
    B1)       need_events; need_llm; run pipeline.metric.therapist_paired ALL; run pipeline.metric.therapist_baseline
              need_anchors; need_testbed2; run pipeline.external.empathy_tactics
              echo "→ out/tables/therapist_llm_ceiling.csv  therapist_density_matched.csv  therapist_baseline.csv  anchor_empathy_tactics.csv" ;;
    B2)       need_events; need_listening; run pipeline.external.listening_extra; run pipeline.metric.multiturn
              echo "→ out/tables/listening_*.csv  seat_strength.csv  script_by_turn.csv  multiturn_extension.csv  out/figures/fig_multiturn.png" ;;
    B3)       need_events; need_testbed2; run pipeline.external.empathy_tactics
              echo "→ out/tables/mint_systems.csv  regex_under_null.csv  stickiness_decomposed.csv  anchor_empathy_tactics.csv" ;;
    B4)       need_events; run pipeline.metric.identification; run pipeline.metric.identification_baselines
              run pipeline.metric.profiles; run pipeline.metric.figure_data; run pipeline.figures.case_study
              ( cd pipeline/figures/handmade/profile_figure && python3 build_figure.py && python3 export_figure.py )
              echo "→ out/tables/identification_*.csv  twin_structure_by_corpus.csv  out/derived/js_fingerprint_distance.csv"
              echo "→ out/figures/fig_transition_lift.png  fig_case_study.png  pipeline/figures/handmade/profile_figure/script_profile_figure.png" ;;
    B5)       need_events; need_robust
              for s in NULLS LEN SHUF CARD; do run pipeline.metric.nulls_ablations "$s"; done
              for r in R1 R2 R3 R4 R5 R6; do run pipeline.metric.annotator_tiers "$r"; done
              run pipeline.metric.annotator_tiers COLLECT; run pipeline.metric.jury; need_anchors
              echo "→ out/tables/bootstrap_ci.csv  alternative_nulls.csv  length_terciles.csv  jury_scripts.csv  loao_tier_gap.csv  anchor_propaganda.csv" ;;
    C1)       need_events; run pipeline.metric.tie_robustness; run pipeline.metric.figure_data; run pipeline.figures.momentum
              echo "→ out/tables/tie_robustness.csv  out/figures/fig_momentum.png" ;;
    C2)       need_events; run pipeline.metric.validate; run pipeline.figures.null_validation; run pipeline.figures.anatomy
              echo "→ out/tables/null_validation.csv  out/figures/fig_null_validation.png  fig_anatomy.png" ;;
    C3)       need_events; need_llm; need_testbed2
              for q in Q1 Q2 Q3 Q4; do run pipeline.metric.sensitivity "$q"; done
              for r in R1 R2 R3 R4 R5 R6; do run pipeline.metric.sensitivity "Q5A:$r"; done
              run pipeline.metric.sensitivity Q5COLLECT; run pipeline.metric.sensitivity_extra ALL
              echo "→ out/tables/bin_sweep.csv  granularity.csv  skipgram.csv  out/tables/reviewer_qs/*.csv" ;;
    C4)       need_events; run pipeline.metric.figure_data; run pipeline.figures.worked_example; run pipeline.figures.method_faithful
              echo "→ out/figures/fig_worked_example.png  fig_method_faithful.png" ;;
    C5)       need_events; need_anchors
              run pipeline.metric.betting_validity V1 --streams=200 --naive=60
              run pipeline.metric.betting_validity V2; run pipeline.metric.betting_validity V3
              run pipeline.metric.betting_applications ALL; run pipeline.figures.betting; run pipeline.figures.explanatory
              echo "→ out/tables/betting_*.csv  reviewer_qs/q7_*.csv  out/figures/fig_betting.png  fig_evidence_ladder.png" ;;
    C6)       need_events; need_testbed2; run pipeline.metric.conformity; run pipeline.metric.quality_matched
              echo "→ out/tables/conformity_*.csv  conformity_summary.json  quality_matched.csv" ;;
    D)        run pipeline.behaviour.attributes; run pipeline.behaviour.spans; run pipeline.behaviour.therapist
              run pipeline.behaviour.dynamics; run pipeline.behaviour.phrases
              run pipeline.figures.behaviour_overview; run pipeline.figures.behaviour_levels
              echo "→ out/tables/behaviour/*.csv  out/figures/fig_behaviour_overview.png  fig_behaviour_language.png" ;;
    appendix) for t in A B1 B2 B3 B4 B5 C1 C2 C3 C4 C5 C6 D; do target $t; done ;;

    figures)  bash paper_figures.sh "${2:-../figures}" ;;
    list)     sed -n '2,12p' "$0" ;;
    *)        echo "unknown target '$1'; try: bash reproduce.sh list" >&2; exit 1 ;;
  esac
}

[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 0; }
for t in "$@"; do target "$t"; done
