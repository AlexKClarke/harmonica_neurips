# Shared settings for the experiment scripts. Override from the environment.
#   DATA         extracted benchmark_v2/ directory (contains ch070/ and ch320/)
#   OUT          where per-seed CSVs are written (default: results/)
#   SEEDS        seed indices of every method (default 0-4); seed i uses FastICA
#                seed 1909+i and network seed 42+i
#   THETA_SEEDS  seed indices of HarmonICA (θ) (default 0-4)
#   PAPER_AS_PRINTED=1  reproduce the printed numbers where the paper pooled
#                seeds: HarmonICA (θ) over seeds 0-9, and the ch070 HarmonICA (PE)
#                row of Table 2 (and the ch070 PE inputs of the figures and
#                per-recording tables) pooling PE seeds 0-4 with θ seeds 5-9
#   JOBS         CPU processes for the Adapt runs (default 1)
#   SKIP_RUNS=1  print the tables from the CSVs already in $OUT, run nothing
: "${DATA:?set DATA to the extracted benchmark_v2/ directory (contains ch070/ and ch320/)}"
OUT=${OUT:-results}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PAPER_AS_PRINTED=${PAPER_AS_PRINTED:-0}
if [[ $PAPER_AS_PRINTED == 1 ]]; then
    THETA_SEEDS=${THETA_SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
else
    THETA_SEEDS=${THETA_SEEDS:-"0 1 2 3 4"}
fi
JOBS=${JOBS:-1}          # parallel CPU shards for bench_adapt.py (adapt())
SKIP_RUNS=${SKIP_RUNS:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
mkdir -p "$OUT"

# Multi-unit methods: bench_baselines.py. Usage: baselines GRID METHOD TAG KWARGS [args...]
baselines() {
    local grid=$1 method=$2 tag=$3 kw=$4; shift 4
    if [[ $SKIP_RUNS == 1 ]]; then return 0; fi
    python "$ROOT/scripts/bench_baselines.py" --data-dir "$DATA/$grid" \
        --val-set "$HERE/lists/${grid}_all.txt" --method "$method" --kwargs "$kw" \
        --tag "$tag" --output "$OUT/${grid}_${tag}.csv" "$@"
}

# Per-unit refiners: bench_sweep.py at R=8. Usage: sweep GRID METHOD TAG KWARGS SEED_IDX [args...]
sweep() {
    local grid=$1 method=$2 tag=$3 kw=$4 i=$5; shift 5
    if [[ $SKIP_RUNS == 1 ]]; then return 0; fi
    python "$ROOT/scripts/bench_sweep.py" --data-dir "$DATA/$grid" \
        --val-set "$HERE/lists/${grid}_all.txt" --method "$method" --kwargs "$kw" \
        --tag "$tag" --output "$OUT/${grid}_${tag}.csv" \
        --ica-seed $((1909 + i)) --nn-seed $((42 + i)) --R 8 "$@"
}

# Adapt rows: bench_adapt.py on the CPU, JOBS shards in parallel, merged into one
# long-format CSV holding every seed. Usage: adapt GRID TAG [bench_adapt.py args...]
adapt() {
    local grid=$1 tag=$2; shift 2
    if [[ $SKIP_RUNS == 1 ]]; then return 0; fi
    local out="$OUT/${grid}_${tag}.csv" seeds="" i pids=() f
    for i in $SEEDS; do seeds+="${seeds:+,}$((1909 + i))"; done
    rm -f "$out" "$out".shard*                # bench_adapt.py appends to --out
    for ((i = 0; i < JOBS; i++)); do
        python "$ROOT/scripts/bench_adapt.py" --data-dir "$DATA" --ch $((10#${grid#ch})) \
            --device cpu --seeds "$seeds" --num-shards "$JOBS" --shard-idx "$i" \
            --out "$out.shard$i" "$@" > "$out.shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i"; done
    for f in "$out".shard*[0-9]; do
        if [[ -s $out ]]; then tail -n +2 "$f" >> "$out"; else cat "$f" > "$out"; fi
    done
    rm -f "$out".shard*[0-9]
}

# Paths of the per-seed CSVs of one tag. Usage: seed_csvs GRID TAG SEED...
seed_csvs() {
    local grid=$1 tag=$2 i; shift 2
    for i in "$@"; do echo "$OUT/${grid}_${tag}_seed$i.csv"; done
}

# Yield table over the per-seed CSVs of one tag that exist for the seeds in
# TABLE_SEEDS (default $SEEDS). Usage: [TABLE_SEEDS=...] table GRID TAG NAME [args...]
table() {
    local grid=$1 tag=$2 name=$3 f files=(); shift 3
    while read -r f; do
        if [[ -f $f ]]; then files+=("$f"); fi
    done < <(seed_csvs "$grid" "$tag" ${TABLE_SEEDS:-$SEEDS})
    if (( ${#files[@]} == 0 )); then
        echo "table: no $OUT/${grid}_${tag}_seed<i>.csv for seeds ${TABLE_SEEDS:-$SEEDS}" >&2
        return 1
    fi
    python "$ROOT/scripts/yield_table.py" --name "$grid  $name" "$@" "${files[@]}"
}
