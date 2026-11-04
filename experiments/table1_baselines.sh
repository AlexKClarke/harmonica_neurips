#!/usr/bin/env bash
# Table 2 baseline rows (Chen, Kramberger, Yeung, Glaser) and the ICA / cBSS
# reference row. The Adapt row is table1_adapt.sh.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table1_baselines.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    for i in $SEEDS; do
        s=$((1909 + i))
        baselines "$grid" chen   "chen_seed$i"   "{\"learning_rate\":0.10,\"cov_forgetting\":0.85,\"ext_fact\":8,\"ica_seed\":$s}"
        baselines "$grid" yeung  "yeung_seed$i"  "{\"z_threshold\":2.0,\"ext_fact\":8,\"ica_seed\":$s}"
        baselines "$grid" glaser_cyclostationary "glaser_seed$i" "{\"R\":8,\"seed\":$s}"
        # ICA / cBSS reference: the FastICA sources scored unrefined (f1_ica).
        sweep "$grid" ica "ica_reference_seed$i" '{}' "$i"
    done
    # Kramberger is deterministic: one run.
    baselines "$grid" kramberger kramberger_seed0 '{"R":8,"calib_window_mode":"time","calib_sil_th":0.8}'
    table "$grid" ica_reference "ICA / cBSS" --column f1_ica
    table "$grid" chen          "Chen 2020"
    table "$grid" kramberger    "Kramberger 2021"
    table "$grid" yeung         "Yeung 2024"
    table "$grid" glaser        "Glaser 2018"
done
