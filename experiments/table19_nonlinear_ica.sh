#!/usr/bin/env bash
# Nonlinear-ICA comparison (linear ICA vs TCL and iVAE), the table promised in
# the review response; it has no table number in the paper yet. Five recordings
# per grid (experiments/lists/nonlinear_ica_ch*.txt), one run each, reported as
# the mean over recordings of ground-truth units recovered at F1 > 0.5.
#   linear ICA, TCL-θ, iVAE-θ   scripts/nonlinear_ica.py --protocol angle, both grids
#   TCL-t (and TCL-θ again)     --protocol time, ch070 only
# TCL-θ depends on how its classifier is trained: the angle protocol prints
# training progress, which leaves the network in eval mode after epoch 50, and
# the time protocol trains it quietly (on the first ch070 recording: 5 and 2
# units). The two runs bracket the range the table quotes for TCL-θ on ch070.
# The ch070 recordings are the ones the table used. The ch320 ones match its
# five pools (N026, N045, N055, N080, N113); the condition of each was not
# recorded, so the SA05-hi recording of each pool is used, as on ch070.
# One GPU, about 6 minutes per recording and protocol.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table19_nonlinear_ica.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
nonlinear_ica() {
    local grid=$1 protocol=$2 tag=$3
    python "$ROOT/scripts/nonlinear_ica.py" --data-dir "$DATA/$grid" \
        --recordings "$HERE/lists/nonlinear_ica_${grid}.txt" --protocol "$protocol" \
        --output "$OUT/${grid}_${tag}.csv"
}
for grid in ${@:-ch070 ch320}; do
    nonlinear_ica "$grid" angle nonlinear_ica
    if [[ $grid == ch070 ]]; then
        nonlinear_ica "$grid" time nonlinear_ica_time
    fi
done
