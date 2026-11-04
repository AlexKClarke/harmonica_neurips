#!/usr/bin/env bash
# HarmonICA (PE) and HarmonICA (θ) rows: main Table 2 (tab:yield_3bin), the
# reference row of main Table 3 / supp. tab:supp:headline_ablation (top-1000
# F1), supp. tab:supp:pe_vs_angle and tab:supp:pe_vs_angle_thresh (mean F1,
# n at F1 >= 0.85 / 0.90 / 0.95) and the r = 3 row of tab:supp:rank. Also prints
# the ground-truth strata of main Table 1 (tab:dataset_strata).
# PE runs over SEEDS and θ over THETA_SEEDS (both 0-4 by default). With
# PAPER_AS_PRINTED=1, θ runs over seeds 0-9 as in Table 2, and two more rows are
# printed: θ over seeds 0-4 (tab:supp:rank, tab:supp:pe_vs_angle_thresh) and, for
# ch070, the PE row as printed in Table 2, which pools PE seeds 0-4 with θ seeds 5-9.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table1_harmonica.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
common() { printf '"epochs":200,"sep_only_warmup":50,"R":8,"ica_seed":%d,"seed":%d' $((1909 + $1)) $((42 + $1)); }
cols=(--top-k 1000 --mean-f1 --also-at 0.85,0.90)
for grid in ${@:-ch070 ch320}; do
    for i in $SEEDS; do
        baselines "$grid" signal_space "harmonica_pe_seed$i" "{$(common "$i"),\"use_angle\":false}"
    done
    for i in $THETA_SEEDS; do
        baselines "$grid" signal_space "harmonica_theta_seed$i" "{$(common "$i"),\"use_angle\":true}"
    done
    mapfile -t pe < <(seed_csvs "$grid" harmonica_pe $SEEDS)
    python "$ROOT/scripts/yield_table.py" --strata --name "$grid  ground-truth units per tier" "${pe[@]}"
    table "$grid" harmonica_pe "HarmonICA (PE)" "${cols[@]}"
    TABLE_SEEDS=$THETA_SEEDS table "$grid" harmonica_theta "HarmonICA (θ)" "${cols[@]}"
    if [[ $PAPER_AS_PRINTED == 1 ]]; then
        TABLE_SEEDS="0 1 2 3 4" table "$grid" harmonica_theta "HarmonICA (θ), seeds 0-4" "${cols[@]}"
        if [[ $grid == ch070 ]]; then
            mapfile -t pooled < <(seed_csvs "$grid" harmonica_pe 0 1 2 3 4; seed_csvs "$grid" harmonica_theta 5 6 7 8 9)
            python "$ROOT/scripts/yield_table.py" --name "$grid  HarmonICA (PE) as printed: PE seeds 0-4 + θ seeds 5-9" \
                "${cols[@]}" "${pooled[@]}"
        fi
    fi
done
