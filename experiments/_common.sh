# Shared settings for the experiment scripts. Override from the environment.
#   DATA   extracted benchmark_v2/ directory (contains ch070/ and ch320/)
#   OUT    where per-seed CSVs are written (default: results/)
#   SEEDS  seed indices; seed i uses FastICA seed 1909+i and network seed 42+i
: "${DATA:?set DATA to the extracted benchmark_v2/ directory (contains ch070/ and ch320/)}"
OUT=${OUT:-results}
SEEDS=${SEEDS:-"0 1 2 3 4"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
mkdir -p "$OUT"

# Multi-unit methods: bench_baselines.py. Usage: baselines GRID METHOD TAG KWARGS [args...]
baselines() {
    local grid=$1 method=$2 tag=$3 kw=$4; shift 4
    python "$ROOT/scripts/bench_baselines.py" --data-dir "$DATA/$grid" \
        --val-set "$HERE/lists/${grid}_all.txt" --method "$method" --kwargs "$kw" \
        --tag "$tag" --output "$OUT/${grid}_${tag}.csv" "$@"
}

# Per-unit refiners: bench_sweep.py at R=8. Usage: sweep GRID METHOD TAG KWARGS SEED_IDX [args...]
sweep() {
    local grid=$1 method=$2 tag=$3 kw=$4 i=$5; shift 5
    python "$ROOT/scripts/bench_sweep.py" --data-dir "$DATA/$grid" \
        --val-set "$HERE/lists/${grid}_all.txt" --method "$method" --kwargs "$kw" \
        --tag "$tag" --output "$OUT/${grid}_${tag}.csv" \
        --ica-seed $((1909 + i)) --nn-seed $((42 + i)) --R 8 "$@"
}

# Yield table over the per-seed CSVs of one tag. Usage: table GRID TAG NAME [args...]
table() {
    local grid=$1 tag=$2 name=$3; shift 3
    python "$ROOT/scripts/yield_table.py" --name "$grid  $name" "$@" "$OUT/${grid}_${tag}"_seed*.csv
}
