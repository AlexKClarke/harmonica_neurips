#!/usr/bin/env bash
# Table 2, Mendez Guerra 2024 (Adapt) row: the adapt_decomp v1.0.0 engine with
# (wh_learning_rate, sv_learning_rate, centroid_momentum) tuned on the
# validation split (experiments/adapt_tuning/, from tune_adapt.sh), plus the
# two default-settings rows (v0 and v1.0.0 engines, no tuning) and the
# ICA / cBSS reference scored inside the same harness (band-pass, calibration
# and scoring of bench_adapt.py), which the supplement compares with the
# Table 2 ICA / cBSS row.
# Runs on the CPU: about 3 CPU-hours for ch070 and 5 for ch320 for the Adapt
# rows, plus 11 and 13 for the harness ICA / cBSS row; JOBS splits it.
# Usage: DATA=/path/to/benchmark_v2 JOBS=16 bash experiments/table1_adapt.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    g=$((10#${grid#ch}))
    adapt "$grid" adapt_tuned --methods adapt_v1_opt_pareto_cov_mvtpe30 \
        --adapt-v1-opt-config "$HERE/adapt_tuning/ch${g}_v1_pareto_cov_mvtpe30/best_params.json"
    adapt "$grid" adapt_default --methods adapt,adapt_v1
    adapt "$grid" ica_harness --methods ica_cbss
    py() { python "$ROOT/scripts/yield_table.py" "$@"; }
    py --name "$grid  Mendez Guerra 2024 (tuned)" --top-k 1000 "$OUT/${grid}_adapt_tuned.csv"
    py --name "$grid  Adapt v0, defaults"   --method adapt    "$OUT/${grid}_adapt_default.csv"
    py --name "$grid  Adapt v1.0.0, defaults" --method adapt_v1 "$OUT/${grid}_adapt_default.csv"
    py --name "$grid  ICA / cBSS, Adapt harness" "$OUT/${grid}_ica_harness.csv"
done
