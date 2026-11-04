#!/usr/bin/env bash
# The twelve Adapt tuning variants. The reported configuration
# (pareto_cov_mvtpe30; tune_adapt.sh, table1_adapt.sh) is one of twelve searches
# on the validation split, all of which were also run on the test split. A
# variant is an objective (pareto: whitening and separation losses, keeping the
# Pareto member with the lowest separation loss; sv: separation loss; roa: rate
# of agreement with ground truth), a pool of calibration units (paired: matched
# to ground truth at RoA >= 0.8; all; sil: SIL >= 0.9; cov: CoV-ISI <= 0.3) and
# a sampler (Adapt's default TPE with 15 start-up trials, or mvtpe30:
# multivariate TPE with 30). 100 trials, seed 1909, one FastICA seed.
# Steps (STEPS, default "tune eval"):
#   tune  rerun each search on the validation split ($VAL), writing
#         $OUT/adapt_tuning/ch<g>_v1_<variant>/. About 1 CPU-hour per search.
#   eval  run each variant's best_params.json on the test split ($DATA) over
#         SEEDS, then print the yield at F1 >= 0.95 per grid and summed over the
#         grids. Reads CONFIGS, by default experiments/adapt_tuning/ (the
#         paper's searches); CONFIGS=$OUT/adapt_tuning uses the tune step's
#         output. About 3 (ch070) and 5 (ch320) CPU-hours per variant; JOBS
#         splits it.
# VARIANTS selects a subset (default: all twelve).
# Usage: VAL=/path/to/benchmark_v2_val DATA=/path/to/benchmark_v2 JOBS=16 \
#        bash experiments/adapt_variants.sh [ch070] [ch320]
set -euo pipefail
STEPS=${STEPS:-"tune eval"}
if [[ " $STEPS " == *" eval "* ]]; then
    source "$(dirname "$0")/_common.sh"
else
    OUT=${OUT:-results}
    HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    ROOT=$(dirname "$HERE")
fi
CONFIGS=${CONFIGS:-$HERE/adapt_tuning}

# variant -> tune_adapt_v1.py search flags
declare -A SEARCH=(
    [pareto_paired]="--approach pareto --pool paired"
    [pareto_all]="--approach pareto --pool all"
    [pareto_sil]="--approach pareto --pool sil --sil-th 0.9"
    [pareto_cov]="--approach pareto --pool cov --cov-th 0.3"
    [pareto_cov_mvtpe30]="--approach pareto --pool cov --cov-th 0.3 --multivariate-tpe --n-startup-trials 30"
    [sv_paired]="--approach sv --pool paired"
    [sv_all]="--approach sv --pool all"
    [sv_sil]="--approach sv --pool sil --sil-th 0.9"
    [sv_cov]="--approach sv --pool cov --cov-th 0.3"
    [sv_cov_mvtpe30]="--approach sv --pool cov --cov-th 0.3 --multivariate-tpe --n-startup-trials 30"
    [roa_paired]="--approach roa --pool paired"
    [roa_all]="--approach roa --pool all"
)
VARIANTS=${VARIANTS:-"pareto_paired pareto_all pareto_sil pareto_cov pareto_cov_mvtpe30
    sv_paired sv_all sv_sil sv_cov sv_cov_mvtpe30 roa_paired roa_all"}
GRIDS=${@:-ch070 ch320}

for v in $VARIANTS; do
    if [[ -z ${SEARCH[$v]:-} ]]; then echo "unknown variant '$v'" >&2; exit 1; fi
done

if [[ " $STEPS " == *" tune "* ]]; then
    : "${VAL:?set VAL to the extracted benchmark_v2_val/ directory (contains ch070/ and ch320/)}"
    for grid in $GRIDS; do
        g=$((10#${grid#ch}))
        for v in $VARIANTS; do
            python "$ROOT/scripts/tune_adapt_v1.py" --data-dir "$VAL" --ch "$g" ${SEARCH[$v]} \
                --n-trials 100 --n-jobs 1 --device cpu --seed 1909 \
                --out-dir "$OUT/adapt_tuning/ch${g}_v1_$v" --verbose
        done
    done
fi

if [[ " $STEPS " == *" eval "* ]]; then
    for grid in $GRIDS; do
        g=$((10#${grid#ch}))
        for v in $VARIANTS; do
            adapt "$grid" "adapt_$v" --methods "adapt_v1_opt_$v" \
                --adapt-v1-opt-config "$CONFIGS/ch${g}_v1_$v/best_params.json"
        done
    done
    for grid in $GRIDS; do
        for v in $VARIANTS; do
            python "$ROOT/scripts/yield_table.py" --name "$grid  $v" "$OUT/${grid}_adapt_$v.csv"
        done
    done
    # Yield at F1 >= 0.95 (mean over seeds, as yield_table.py), summed over the grids.
    python - "$OUT" "$GRIDS" $VARIANTS <<'PY'
import sys
import pandas as pd

out, grids, variants = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
rows = []
for v in variants:
    y = {}
    for grid in grids:
        df = pd.read_csv(f"{out}/{grid}_adapt_{v}.csv")
        df = df[df["ok"].astype(bool)]
        y[grid] = df.groupby("seed_idx")["best_f1"].apply(lambda f: (f >= 0.95).sum()).mean()
    rows.append((v, y, sum(y.values())))
print(f"\n{'variant':20s}" + "".join(f"{g:>8s}" for g in grids) + f"{'sum':>8s}")
for v, y, total in sorted(rows, key=lambda r: -r[2]):
    print(f"{v:20s}" + "".join(f"{y[g]:8.1f}" for g in grids) + f"{total:8.1f}")
PY
fi
