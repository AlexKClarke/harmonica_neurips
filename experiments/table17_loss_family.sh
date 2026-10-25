#!/usr/bin/env bash
# Supp. Table 17: stationarity-loss family, filter-space refiner (refine_harmonica_style).
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table17_loss_family.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
LOSSES="gmm negentropy_g1 negentropy_g2 kurtosis power_p"
for grid in ${@:-ch070 ch320}; do
    for loss in $LOSSES; do
        for i in $SEEDS; do
            sweep "$grid" harmonica_style "loss_${loss}_seed$i" "{\"comp_loss\":\"$loss\"}" "$i"
        done
    done
    for loss in $LOSSES; do
        table "$grid" "loss_$loss" "$loss" --top-k 1000
    done
done
