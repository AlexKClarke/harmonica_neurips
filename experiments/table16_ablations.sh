#!/usr/bin/env bash
# Supp. Table 16 ablation rows. Compare against HarmonICA (PE) from table1_harmonica.sh.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/table16_ablations.sh [ch070] [ch320]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
for grid in ${@:-ch070 ch320}; do
    for i in $SEEDS; do
        common=$(printf '"epochs":200,"sep_only_warmup":50,"R":8,"ica_seed":%d,"seed":%d' $((1909 + i)) $((42 + i)))
        baselines "$grid" signal_space "abl_coldstart_seed$i" "{$common,\"warmstart_w\":false}"
        baselines "$grid" signal_space "abl_additive_seed$i"  "{$common,\"config_overrides\":{\"compensation_model\":\"affine\"}}"
        baselines "$grid" signal_space "abl_coupled_seed$i"   "{$common,\"couple_grads\":true}"
        # No peel-off: per-unit HarmonICA, each unit refined from its best-matching ICA source.
        sweep "$grid" signal_space    "abl_nopeel_seed$i"   '{"epochs":200,"sep_only_warmup":50}' "$i"
        # Kurtosis loss: filter-space refiner (also the kurtosis row of supp. Table 17).
        sweep "$grid" harmonica_style "abl_kurtosis_seed$i" '{"comp_loss":"kurtosis"}' "$i"
    done
    for tag in abl_coldstart abl_additive abl_coupled abl_nopeel abl_kurtosis; do
        table "$grid" "$tag" "$tag" --top-k 1000
    done
done
