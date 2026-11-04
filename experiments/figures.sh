#!/usr/bin/env bash
# Main Fig. 3 (fig:pareto_recall, yield x recall Pareto), the supplement's
# yield x F1 Pareto, the recall crossovers of tab:supp:pe_vs_angle_pareto, the
# two overlap figures, the per-recording tables (tab:supp:detailed_yield,
# tab:supp:yield_per_rec, tab:supp:seed_std), the significance tables
# (tab:supp:bootstrap, tab:supp:wilcoxon) and the runtimes of tab:supp:impl,
# from the per-seed CSVs that table1_harmonica.sh, table1_baselines.sh and
# table1_adapt.sh write to $OUT.
#
# HarmonICA inputs: PE over SEEDS, θ over THETA_SEEDS. With PAPER_AS_PRINTED=1
# (run table1_harmonica.sh with it first) they are the printed sets: θ over
# seeds 0-9, and the ch070 HarmonICA curve pools PE seeds 0-4 with θ seeds 5-9.
# The per-recording and significance tables are then also written with that
# pooled set to $FIG/as_printed/: the paper's ch070 HarmonICA cells of the
# per-recording tables, its bootstrap mean and its Wilcoxon Delta come from
# there, the Wilcoxon p-values and wins / losses / ties from $FIG.
# Other choices of the original analysis, kept so the outputs match the paper:
#   - the Pareto curves pool both Kramberger runs (the default calibration below
#     and the paper's time-window calibration); the overlap figures use the
#     default one only, the tables the paper's one;
#   - the overlap heatmap uses the segmented Glaser (run_glaser), the stacked
#     bars and everything else the cyclostationary one;
#   - the overlap figures use HarmonICA (PE) over SEEDS in both modes;
#   - the printed Glaser p-values of tab:supp:wilcoxon are one-sided
#     (p_greater in wilcoxon.csv), the others two-sided (p).
# The ICA / cBSS reference scored inside the Adapt harness (table1_adapt.sh) is
# tabulated in the Pareto crossovers as ica_harness when its CSVs exist.
# Usage: DATA=/path/to/benchmark_v2 bash experiments/figures.sh   (writes $FIG, default $OUT/figures)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
F="$ROOT/scripts/figures"
FIG=${FIG:-$OUT/figures}
mkdir -p "$FIG"

# Per-seed CSVs of one tag on both grids. Usage: csvs TAG [SEED...] (default $SEEDS)
csvs() { local tag=$1 g; shift; for g in ch070 ch320; do seed_csvs "$g" "$tag" ${@:-$SEEDS}; done; }
mapfile -t H_PE < <(csvs harmonica_pe)
mapfile -t H_THETA < <(csvs harmonica_theta $THETA_SEEDS)
mapfile -t ICA < <(csvs ica_reference)
mapfile -t CHEN < <(csvs chen)
mapfile -t YEUNG < <(csvs yeung)
mapfile -t GLASER < <(csvs glaser)
mapfile -t GLASER_SEG < <(csvs glaser_segmented)
mapfile -t KRAM < <(csvs kramberger 0)
mapfile -t KRAM_DEFAULT < <(csvs kramberger_default 0)
ADAPT=("$OUT/ch070_adapt_tuned.csv" "$OUT/ch320_adapt_tuned.csv")
H_CURVE=("${H_PE[@]}")
if [[ $PAPER_AS_PRINTED == 1 ]]; then
    mapfile -t H_CURVE < <(seed_csvs ch070 harmonica_pe 0 1 2 3 4; seed_csvs ch070 harmonica_theta 5 6 7 8 9
                           seed_csvs ch320 harmonica_pe $SEEDS)
fi
HARNESS=()
if [[ -f $OUT/ch070_ica_harness.csv && -f $OUT/ch320_ica_harness.csv ]]; then
    HARNESS=(--input ica_harness "$OUT/ch070_ica_harness.csv" "$OUT/ch320_ica_harness.csv")
fi
missing=0
for f in "${H_CURVE[@]}" "${H_PE[@]}" "${H_THETA[@]}" "${ICA[@]}" "${CHEN[@]}" "${YEUNG[@]}" \
         "${GLASER[@]}" "${KRAM[@]}" "${ADAPT[@]}"; do
    if [[ ! -f $f ]]; then echo "figures.sh: missing $f" >&2; missing=1; fi
done
(( missing == 0 )) || exit 1

# Two extra baseline runs that only the figures use.
for grid in ch070 ch320; do
    [[ -f $OUT/${grid}_kramberger_default_seed0.csv ]] ||
        baselines "$grid" kramberger kramberger_default_seed0 '{"R":8}'
    for i in $SEEDS; do
        [[ -f $OUT/${grid}_glaser_segmented_seed$i.csv ]] ||
            baselines "$grid" glaser "glaser_segmented_seed$i" \
                "{\"min_overlap_f1\":0.4,\"sil_th\":0.9,\"R\":8,\"seed\":$((1909 + i))}"
    done
done

for axis in recall f1; do
    python "$F/pareto.py" --axis "$axis" --out-dir "$FIG" \
        --input harmonica "${H_CURVE[@]}" \
        --input ica "${ICA[@]}" \
        --input chen "${CHEN[@]}" \
        --input yeung "${YEUNG[@]}" \
        --input kramberger "${KRAM[@]}" "${KRAM_DEFAULT[@]}" \
        --input glaser "${GLASER[@]}" \
        --input adapt "${ADAPT[@]}" \
        --input harmonica_theta "${H_THETA[@]}" ${HARNESS[@]+"${HARNESS[@]}"}
done

common=(--input harmonica "${H_PE[@]}"
        --input ica "${ICA[@]}"
        --input chen "${CHEN[@]}"
        --input kramberger "${KRAM_DEFAULT[@]}"
        --input yeung "${YEUNG[@]}"
        --input adapt "${ADAPT[@]}")
python "$F/overlap.py" heatmap --out-dir "$FIG" "${common[@]}" --input glaser "${GLASER_SEG[@]}"
python "$F/overlap.py" bars --out-dir "$FIG" "${common[@]}" --input glaser "${GLASER[@]}"

baselines_in=(--input ica "${ICA[@]}"
              --input chen "${CHEN[@]}"
              --input kramberger "${KRAM[@]}"
              --input yeung "${YEUNG[@]}"
              --input glaser "${GLASER[@]}"
              --input adapt "${ADAPT[@]}")
python "$F/stats.py" --out-dir "$FIG" --input harmonica "${H_PE[@]}" "${baselines_in[@]}"
if [[ $PAPER_AS_PRINTED == 1 ]]; then
    python "$F/stats.py" --out-dir "$FIG/as_printed" --input harmonica "${H_CURVE[@]}" "${baselines_in[@]}"
fi

# tab:supp:impl runtimes: wall time per (recording, seed) run. The ICA / cBSS
# reference run (bench_sweep.py --method ica) times preprocessing, FastICA and
# scoring only; Adapt runs on one CPU thread.
runtime() { python "$ROOT/scripts/yield_table.py" --runtime --name "$@"; }
for grid in ch070 ch320; do
    for spec in "HarmonICA (PE):harmonica_pe" "ICA / cBSS reference run:ica_reference" \
                "Chen 2020:chen" "Yeung 2024:yeung" "Glaser 2018:glaser"; do
        mapfile -t f < <(seed_csvs "$grid" "${spec#*:}" $SEEDS)
        runtime "$grid  ${spec%:*}" "${f[@]}"
    done
    runtime "$grid  Kramberger 2021" "$OUT/${grid}_kramberger_seed0.csv"
    runtime "$grid  Mendez Guerra 2024" "$OUT/${grid}_adapt_tuned.csv"
done
