#!/usr/bin/env bash
# Supp. Table 19: identifiable nonlinear ICA (TCL, iVAE). The paper reports mean
# motor units per recording at F1 > 0.5: divide the printed totals by the number
# of recordings (168 per grid). These methods are seed-sensitive; expect ranges.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table19_nonlinear_ica.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    baselines "$grid" tcl  tcl_angle_seed0 '{}'
    baselines "$grid" tcl  tcl_time_seed0  '{}' --angle-mode none
    baselines "$grid" ivae ivae_angle_seed0 '{}'
    for tag in tcl_angle tcl_time ivae_angle; do
        table "$grid" "$tag" "$tag" --threshold 0.5
    done
done
