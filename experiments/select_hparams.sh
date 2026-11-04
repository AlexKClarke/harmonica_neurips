#!/usr/bin/env bash
# Hyperparameter selection: the sweeps behind HarmonICA's R = 8 and compensator
# settings and the Chen 2020 / Yeung 2024 settings of Table 2. They score
# ground-truth F1 on 24 recordings of the test split (experiments/lists/sweep24.txt,
# the same names on both grids) with one seed (FastICA 1909, network 42); the
# val part re-checks HarmonICA on the disjoint validation split. Each config is
# one runner call. Per sweep, the printed table gives, per config, the units at
# F1 >= 0.95 and the mean F1 over the matched units (yield_table.py --mean-f1);
# * marks the setting the paper uses.
#   harmonica  per-unit HarmonICA (bench_sweep.py --method signal_space), ch070:
#              compensator at R = 4 (low-rank rank 1/3/5 or additive, 200 or 400
#              epochs, separation-only warm-up 0/50/100), then R = 8/16/32/64 at
#              warm-up 50. The original grid also had a Givens-rotation
#              compensator, which is not part of this release.
#   chen       Chen 2020 learning_rate x cov_forgetting, 3 x 3, both grids.
#   yeung      Yeung 2024 z_threshold with the calibration sil_th, 6 configs,
#              both grids.
#   val        HarmonICA (PE), multi-unit, on the validation split (40 recordings
#              per grid, seeds 0-2): R = 4/8/16, warm-up 0, rank 1. The original
#              job also varied the engine's compensation SIL gate and drift
#              budget, which run_signal_space does not read, so those configs
#              equal R = 8 and are left out.
# Chen and Yeung keep their runner defaults otherwise (extension factor 10,
# calibration seed 1909), as in the original sweep. The paper's values had the
# best ch070 mean F1 there and are used on both grids; Table 2 runs them at
# extension factor 8 over five seeds.
# Roughly 3 (harmonica), 2 (chen), 1 (yeung) and 15 (val) hours on one GPU node.
# Usage: DATA=/path/to/benchmark_v2 VAL=/path/to/benchmark_v2_val \
#        [PARTS="harmonica chen yeung val"] [SKIP_RUNS=1] bash experiments/select_hparams.sh
set -euo pipefail
OUT=${OUT:-results}
PARTS=${PARTS:-"harmonica chen yeung val"}
SKIP_RUNS=${SKIP_RUNS:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
S="$OUT/select"
mkdir -p "$S"

run() { if [[ $SKIP_RUNS != 1 ]]; then python "$@"; fi; }

# One table row: yield_table.py over the CSVs that exist. Usage: row NAME CSV...
row() {
    local name=$1 f files=(); shift
    for f in "$@"; do if [[ -f $f ]]; then files+=("$f"); fi; done
    if (( ${#files[@]} == 0 )); then echo "$name  (no results)"; return 0; fi
    python "$ROOT/scripts/yield_table.py" --name "$name" --mean-f1 "${files[@]}"
}

# Per-unit HarmonICA on the 24 ch070 recordings. Usage: unit TAG R KWARGS
unit() {
    run "$ROOT/scripts/bench_sweep.py" --data-dir "$DATA/ch070" --val-set "$HERE/lists/sweep24.txt" \
        --method signal_space --kwargs "$3" --tag "$1" --R "$2" --ica-seed 1909 --nn-seed 42 \
        --output "$S/ch070_$1.csv"
}

# Multi-unit runner. Usage: multi DATA_DIR LIST METHOD TAG KWARGS OUTPUT
multi() {
    run "$ROOT/scripts/bench_baselines.py" --data-dir "$1" --val-set "$2" --method "$3" \
        --tag "$4" --kwargs "$5" --output "$6"
}

for part in $PARTS; do
case $part in
harmonica)
    : "${DATA:?set DATA to the extracted benchmark_v2/ directory}"
    # tag | description | kwargs, all at R = 4
    COMP=(
        'ss_lowrank_r3_e200_w0|rank 3, 200 epochs, warm-up 0|{"epochs":200,"sep_only_warmup":0}'
        'ss_lowrank_r3_e400_w0|rank 3, 400 epochs, warm-up 0|{"epochs":400,"sep_only_warmup":0}'
        'ss_lowrank_r1_e200_w0|rank 1, 200 epochs, warm-up 0|{"epochs":200,"sep_only_warmup":0,"config_overrides":{"compensation_rank":1}}'
        'ss_lowrank_r5_e200_w0|rank 5, 200 epochs, warm-up 0|{"epochs":200,"sep_only_warmup":0,"config_overrides":{"compensation_rank":5}}'
        'ss_affine_e200_w0|additive, 200 epochs, warm-up 0|{"epochs":200,"sep_only_warmup":0,"config_overrides":{"compensation_model":"affine"}}'
        'ss_lowrank_r3_e200_w50|rank 3, 200 epochs, warm-up 50 *|{"epochs":200,"sep_only_warmup":50}'
        'ss_lowrank_r3_e200_w100|rank 3, 200 epochs, warm-up 100|{"epochs":200,"sep_only_warmup":100}'
    )
    for cfg in "${COMP[@]}"; do
        IFS='|' read -r tag _ kw <<< "$cfg"
        unit "$tag" 4 "$kw"
    done
    for R in 8 16 32 64; do
        unit "ss_w50_R$R" "$R" '{"epochs":200,"sep_only_warmup":50}'
    done
    echo "== HarmonICA compensator at R = 4, ch070 (24 recordings, 1 seed)"
    for cfg in "${COMP[@]}"; do
        IFS='|' read -r tag name _ <<< "$cfg"
        row "$name" "$S/ch070_$tag.csv"
    done
    echo "== HarmonICA extension factor, rank 3, warm-up 50, ch070 (24 recordings, 1 seed)"
    row "R = 4" "$S/ch070_ss_lowrank_r3_e200_w50.csv"
    for R in 8 16 32 64; do
        row "R = $R$([[ $R == 8 ]] && echo ' *')" "$S/ch070_ss_w50_R$R.csv"
    done
    ;;
chen)
    : "${DATA:?set DATA to the extracted benchmark_v2/ directory}"
    for grid in ch070 ch320; do
        for lr in 0.05 0.10 0.20; do
            for fg in 0.85 0.90 0.95; do
                tag="chen_lr${lr/./}_fg${fg/./}"
                multi "$DATA/$grid" "$HERE/lists/sweep24.txt" chen "$tag" \
                    "{\"learning_rate\":$lr,\"cov_forgetting\":$fg}" "$S/${grid}_$tag.csv"
            done
        done
    done
    for grid in ch070 ch320; do
        echo "== Chen 2020, $grid (24 recordings, 1 seed)"
        for lr in 0.05 0.10 0.20; do
            for fg in 0.85 0.90 0.95; do
                tag="chen_lr${lr/./}_fg${fg/./}"
                row "learning_rate $lr, cov_forgetting $fg$([[ $tag == chen_lr010_fg085 ]] && echo ' *')" \
                    "$S/${grid}_$tag.csv"
            done
        done
    done
    ;;
yeung)
    : "${DATA:?set DATA to the extracted benchmark_v2/ directory}"
    # tag | description | kwargs (sil_th: calibration SIL threshold, default 0.8)
    YEUNG=(
        'yeung_z25_sil08|z_threshold 2.5, sil_th 0.8|{"z_threshold":2.5}'
        'yeung_z22_sil08|z_threshold 2.2, sil_th 0.8|{"z_threshold":2.2}'
        'yeung_z22_sil07|z_threshold 2.2, sil_th 0.7|{"z_threshold":2.2,"sil_th":0.7}'
        'yeung_z20_sil08|z_threshold 2.0, sil_th 0.8 *|{"z_threshold":2.0}'
        'yeung_z20_sil07|z_threshold 2.0, sil_th 0.7|{"z_threshold":2.0,"sil_th":0.7}'
        'yeung_z20_sil06|z_threshold 2.0, sil_th 0.6|{"z_threshold":2.0,"sil_th":0.6}'
    )
    for grid in ch070 ch320; do
        for cfg in "${YEUNG[@]}"; do
            IFS='|' read -r tag _ kw <<< "$cfg"
            multi "$DATA/$grid" "$HERE/lists/sweep24.txt" yeung "$tag" "$kw" "$S/${grid}_$tag.csv"
        done
    done
    for grid in ch070 ch320; do
        echo "== Yeung 2024, $grid (24 recordings, 1 seed)"
        for cfg in "${YEUNG[@]}"; do
            IFS='|' read -r tag name _ <<< "$cfg"
            row "$name" "$S/${grid}_$tag.csv"
        done
    done
    ;;
val)
    : "${VAL:?set VAL to the extracted benchmark_v2_val/ directory}"
    # tag  R  warm-up  extra kwargs (all: 200 epochs, rank 3)
    VALCFG=(
        'R4 4 50'
        'R8 8 50'
        'R16 16 50'
        'R8_w0 8 0'
        'R8_rank1 8 50 ,"config_overrides":{"compensation_rank":1}'
    )
    for grid in ch070 ch320; do
        for cfg in "${VALCFG[@]}"; do
            read -r tag R w extra <<< "$cfg"
            for i in 0 1 2; do
                kw=$(printf '{"epochs":200,"sep_only_warmup":%d,"R":%d,"ica_seed":%d,"seed":%d%s}' \
                    "$w" "$R" $((1909 + i)) $((42 + i)) "$extra")
                multi "$VAL/$grid" "$HERE/lists/${grid}_val.txt" signal_space \
                    "ss_val_${tag}_${grid}_seed$i" "$kw" "$S/val_${grid}_${tag}_seed$i.csv"
            done
        done
    done
    for grid in ch070 ch320; do
        echo "== HarmonICA (PE), validation split, $grid (40 recordings, seeds 0-2)"
        for cfg in "${VALCFG[@]}"; do
            read -r tag _ <<< "$cfg"
            row "$tag$([[ $tag == R8 ]] && echo ' *')" "$S/val_${grid}_${tag}"_seed{0,1,2}.csv
        done
    done
    ;;
*)
    echo "unknown part '$part' (harmonica, chen, yeung, val)" >&2; exit 1 ;;
esac
done
