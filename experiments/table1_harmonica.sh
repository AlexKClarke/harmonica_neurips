#!/usr/bin/env bash
# Table 1 / supp. Table 14: HarmonICA (PE) and HarmonICA (θ), 5 seeds per grid.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table1_harmonica.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    for i in $SEEDS; do
        common=$(printf '"epochs":200,"sep_only_warmup":50,"R":8,"ica_seed":%d,"seed":%d' $((1909 + i)) $((42 + i)))
        baselines "$grid" signal_space "harmonica_pe_seed$i"    "{$common,\"use_angle\":false}"
        baselines "$grid" signal_space "harmonica_theta_seed$i" "{$common,\"use_angle\":true}"
    done
    table "$grid" harmonica_pe    "HarmonICA (PE)" --top-k 1000
    table "$grid" harmonica_theta "HarmonICA (θ)"  --top-k 1000
done
