#!/usr/bin/env bash
# Hyperparameter search behind the Adapt row of Table 2, on the validation split
# (40 recordings per grid, disjoint subjects from the test set). Multivariate
# TPE, 30 random start-up trials, 100 trials, seed 1909; objectives are Adapt's
# own whitening and separation losses on the calibration units with CoV-ISI <= 0.3,
# and the Pareto member with the lowest separation loss is kept. No ground truth
# enters the objective. ~1 CPU-hour per grid. experiments/adapt_tuning/ holds
# the paper's output; this writes to $OUT/adapt_tuning/.
# Usage: VAL=/path/to/benchmark_v2_val bash experiments/tune_adapt.sh [ch070] [ch320]
set -euo pipefail
: "${VAL:?set VAL to the extracted benchmark_v2_val/ directory (contains ch070/ and ch320/)}"
OUT=${OUT:-results}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for grid in ${@:-ch070 ch320}; do
    g=$((10#${grid#ch}))
    python "$ROOT/scripts/tune_adapt_v1.py" --data-dir "$VAL" --ch "$g" \
        --approach pareto --pool cov --cov-th 0.3 --multivariate-tpe --n-startup-trials 30 \
        --n-trials 100 --n-jobs 1 --device cpu --seed 1909 \
        --out-dir "$OUT/adapt_tuning/ch${g}_v1_pareto_cov_mvtpe30" --verbose
done
