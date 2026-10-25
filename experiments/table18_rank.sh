#!/usr/bin/env bash
# Supp. Table 18: compensator rank sweep (angle-conditioned). The r = 3 row is
# HarmonICA (θ) from table1_harmonica.sh.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table18_rank.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    for rank in 5 7; do
        for i in $SEEDS; do
            common=$(printf '"epochs":200,"sep_only_warmup":50,"R":8,"ica_seed":%d,"seed":%d' $((1909 + i)) $((42 + i)))
            baselines "$grid" signal_space "rank${rank}_seed$i" "{$common,\"use_angle\":true,\"config_overrides\":{\"compensation_rank\":$rank}}"
        done
        table "$grid" "rank$rank" "rank $rank"
    done
done
