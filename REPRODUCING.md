# Reproducing the paper

This file maps every table and figure of the paper and its supplement to the
code that produces it. Main-paper items use the compiled numbering (Fig. 1–3,
Table 1–3); supplementary items are named by their LaTeX label.
[`MANIFEST.md`](MANIFEST.md) lists every file in the release and the results it
serves.

## How the pieces fit

Four runners write per-recording CSVs:

- `scripts/bench_baselines.py`: multi-unit methods (HarmonICA and the
  baselines). One row per ground-truth unit that a source matched: the
  best-matching source and its precision, recall and F1.
- `scripts/bench_sweep.py`: per-unit refiners. Each ground-truth unit is
  refined from its best-matching FastICA source (an oracle pairing), and the
  `*_ica` columns score the unrefined source. `--method ica` skips refinement;
  this is the ICA / cBSS reference row.
- `scripts/bench_adapt.py`: the Mendez Guerra 2024 (Adapt) rows and an ICA /
  cBSS reference scored in the same harness. One long-format CSV holds every
  seed.
- `scripts/nonlinear_ica.py`: linear ICA, TCL and iVAE on a short list of
  recordings.

`scripts/yield_table.py` turns these CSVs into table cells: the units with
F1 ≥ 0.95, binned by the recording's pool size (`n_gt` < 40, 40–69, ≥ 70),
as mean ± s.d. across seeds. Optional columns give the mean F1 of the best
1000 units (`--top-k 1000`, "top-1000 F1"), the mean F1 over every unit
(`--mean-f1`) and the yield at other thresholds (`--also-at`). `--strata`
prints Table 1 and `--runtime` prints wall times. The paper prints each yield
total as the sum of its rounded tier cells; `yield_table.py` prints the
unrounded mean. `scripts/figures/` holds the Pareto and overlap figures and
the per-recording and significance tables.

Each `experiments/*.sh` script runs its methods over seeds and recordings,
then prints its table rows.

## Running

```bash
export DATA=/path/to/benchmark_v2 OUT=results
bash experiments/table1_harmonica.sh            # GPU
bash experiments/table1_baselines.sh            # CPU
JOBS=16 bash experiments/table1_adapt.sh        # CPU
bash experiments/figures.sh                     # reads the CSVs of the three scripts above
```

`figures.sh` needs the CSVs of the three `table1_*.sh` scripts, run with the
same seed settings. The ablation, loss, rank and nonlinear-ICA scripts are
independent of each other and of `figures.sh`.

Settings (`experiments/_common.sh`), all read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `DATA` | required | the extracted `benchmark_v2/` directory, which holds `ch070/` and `ch320/` |
| `VAL` | none | the extracted `benchmark_v2_val/`; only `tune_adapt.sh`, `adapt_variants.sh` and `select_hparams.sh` read it |
| `OUT` | `results` | where the CSVs are written |
| `SEEDS` | `0 1 2 3 4` | seed indices of every method |
| `THETA_SEEDS` | `0 1 2 3 4`; `0`–`9` with `PAPER_AS_PRINTED=1` | seed indices of HarmonICA (θ) |
| `PAPER_AS_PRINTED` | `0` | `1` reproduces the seed sets the paper printed (next section) |
| `JOBS` | `1` | parallel CPU processes for the Adapt runs |
| `SKIP_RUNS` | `0` | `1` prints the tables from the CSVs already in `$OUT` and runs nothing |
| `FIG` | `$OUT/figures` | output directory of `figures.sh` |

Seed index `i` means FastICA seed `1909 + i` and network seed `42 + i`.

The scripts loop over seeds and recordings serially. On an A40, HarmonICA
takes about 45 s per recording, so one seed of one grid takes about 2
GPU-hours. The Adapt rows take about 3 (ch070) and 5 (ch320) CPU-hours, and
the harness ICA / cBSS row another 11 and 13; `JOBS` splits that across
processes. To parallelise the GPU work, run one seed and grid per job, then
print the tables in one pass:

```bash
SEEDS=0 THETA_SEEDS=0 bash experiments/table1_harmonica.sh ch320   # one job per seed and grid
SKIP_RUNS=1 bash experiments/table1_harmonica.sh                    # then print the tables
```

## Seed sets: the default and `PAPER_AS_PRINTED=1`

By default every method runs seed indices 0–4. The paper's numbers use
these sets except in two places, which `PAPER_AS_PRINTED=1` reproduces:

1. **HarmonICA (θ) averages seeds 0–9** in Table 2, `tab:supp:pe_vs_angle`
   and `tab:supp:pe_vs_angle_pareto`. `tab:supp:rank` (the r = 3 row) and
   `tab:supp:pe_vs_angle_thresh` use θ seeds 0–4.
2. **The ch070 HarmonICA (PE) results pool PE seeds 0–4 with θ seeds 5–9.**
   This applies to the ch070 PE row of Table 2 and `tab:supp:pe_vs_angle`, and
   to the yield of the reference row of Table 3 and
   `tab:supp:headline_ablation`. It also applies to the ch070 n at F1 ≥ 0.95
   of `tab:supp:pe_vs_angle_thresh` and to the ch070 HarmonICA Pareto curves:
   the ch070 versions of Fig. 3 and `fig:supp:pareto_f1_ch320`, the ch070
   ratios quoted in the text and the ch070 PE row of
   `tab:supp:pe_vs_angle_pareto`. Finally, it applies to the ch070 HarmonICA
   cells of `tab:supp:detailed_yield`, `tab:supp:yield_per_rec` and
   `tab:supp:seed_std`, the bootstrap mean of `tab:supp:bootstrap` and the
   Δ column of `tab:supp:wilcoxon`. ch320 PE always uses PE seeds 0–4.

| Row (yield at F1 ≥ 0.95, total) | Default | `PAPER_AS_PRINTED=1` | Printed |
|---|---|---|---|
| HarmonICA (PE), ch070 | 449.6 (PE seeds 0–4) | 475.8 (PE 0–4 + θ 5–9) | 476 |
| HarmonICA (PE), ch320 | 514.4 | 514.4 | 515 |
| HarmonICA (θ), ch070 | 501.8 (seeds 0–4) | 501.9 (seeds 0–9) | 502 |
| HarmonICA (θ), ch320 | 566.0 (seeds 0–4) | 571.8 (seeds 0–9) | 571 |

(Printed totals are sums of rounded tier cells, e.g. 96 + 263 + 156 = 515.)

What `PAPER_AS_PRINTED=1` changes:

- `table1_harmonica.sh` also trains θ seeds 5–9 on both grids (10 more
  seed-grid runs). It prints θ over seeds 0–9 and θ over seeds 0–4. For
  ch070 it also prints "HarmonICA (PE) as printed: PE seeds 0–4 + θ seeds
  5–9".
- `figures.sh` takes θ over seeds 0–9 and draws the ch070 HarmonICA curve
  from the pooled set. It also writes a second set of per-recording and
  significance tables to `$FIG/as_printed/`. Use that directory for the
  ch070 HarmonICA cells of the per-recording tables, the bootstrap mean and
  the Wilcoxon Δ. Read the Wilcoxon p-values and wins / losses / ties from
  `$FIG`, which uses PE seeds 0–4.
- Every other script ignores it. `table18_rank.sh` takes its r = 3 row from
  the θ seeds 0–4 row of `table1_harmonica.sh`.

Run `table1_harmonica.sh` and `figures.sh` with the same setting.

## Main paper

| Result | Script | Method, settings and where to read it |
|---|---|---|
| Fig. 1 (`fig:motivation`) | none | Illustration, no plotting code. The figure was assembled by hand, and the recording and unit behind panels (a)–(b) were not recorded. Panel (d)'s claim about the losses is quantified in `tab:supp:loss`. |
| Fig. 2 (`fig:training`), Algorithm 1 | none | Schematic and pseudocode; no numbers. Implemented by `run_signal_space` (`src/harmonica/baselines/signal_space.py`) and `src/harmonica/models/engine/`. |
| Table 1 (`tab:dataset_strata`) | `table1_harmonica.sh` | Block "ground-truth units per tier": `yield_table.py --strata` on the HarmonICA (PE) CSVs. It is identical on both grids. |
| Table 2 (`tab:yield_3bin`), HarmonICA (PE) | `table1_harmonica.sh` | `run_signal_space` with `R=8`, `epochs=200`, `sep_only_warmup=50`, `use_angle=false`. The compensator is conditioned on a sinusoidal positional encoding of time. For the seed sets, see above. |
| Table 2, HarmonICA (θ) | `table1_harmonica.sh` | As above with `use_angle=true`: the joint angle conditions the compensator. |
| Table 2, ICA / cBSS | `table1_baselines.sh` | `bench_sweep.py --method ica` at `R=8`, `f1_ica` column: the FastICA sources, unrefined. The extraction budget is `min(3 × n_gt, 108)`. |
| Table 2, Chen 2020 | `table1_baselines.sh` | `run_chen`: `learning_rate=0.10`, `cov_forgetting=0.85`, `ext_fact=8` |
| Table 2, Kramberger 2021 | `table1_baselines.sh` | `run_kramberger`: one run (the method takes no seed); `R=8`, time-window calibration, `calib_sil_th=0.8` |
| Table 2, Yeung 2024 | `table1_baselines.sh` | `run_yeung`: `z_threshold=2.0`, `ext_fact=8` |
| Table 2, Glaser 2018 | `table1_baselines.sh` | `run_glaser_cyclostationary` (cyclostationary CKC with PNR-gated tracking), `R=8` |
| Table 2, Mendez Guerra 2024 | `table1_adapt.sh` | `bench_adapt.py --methods adapt_v1_opt_pareto_cov_mvtpe30`: the adapt_decomp v1.0.0 engine with its NeuroMotion configuration. The exceptions are `wh_learning_rate`, `sv_learning_rate` and `centroid_momentum`, taken from `experiments/adapt_tuning/ch{70,320}_v1_pareto_cov_mvtpe30/best_params.json`. Recordings are band-passed at 10–1000 Hz. Calibration runs the shared Toeplitz (`L=8`) → PCA (128) → ZCA → FastICA (SIL ≥ 0.8) pipeline on the first 1.5 s. Runs on the CPU. |
| Table 2 caption: paired Wilcoxon p-values | `figures.sh` | `wilcoxon.csv`, column `p` (two-sided), HarmonICA (PE) seeds 0–4 against each baseline |
| Fig. 3 (`fig:pareto_recall`) and its in-text ratios | `figures.sh` | `pareto.py --axis recall` writes `pareto_recall_ch320.pdf` (the figure), plus the `_ch070` and `_combined` versions. The units per recording-seed at recall ≥ 0.99 / 0.95 / 0.90 / 0.80 are in `pareto_recall_crossovers.csv`, and each HarmonICA variant's smallest margin over every drawn baseline at every threshold is in `pareto_recall_dominance.txt`. The ch070 ratios need `PAPER_AS_PRINTED=1`. |
| Table 3 (`tab:ablation`) | `table16_ablations.sh`, `table1_harmonica.sh` | `table16_ablations.sh` prints the ablation rows (top-1000 F1, yield). The reference row is HarmonICA (PE) from `table1_harmonica.sh`: its top-1000 F1 and ± s.d. are PE seeds 0–4. Its printed ch070 yield is the Table 2 value (476, pooled), and its ch320 yield is the sum of the rounded Table 2 cells (515). Settings are listed under `tab:supp:headline_ablation`. |
| In-text discovery claim (units the FastICA seed does not accept) | `figures.sh` | `fig:supp:overlap_bars` (below) |
| Checklist: runtimes and compute | `figures.sh` | Runtime lines at the end of the output (see `tab:supp:impl`). The compute budget is arithmetic on these. |

## Supplement

### Background and benchmark

| Result | Script | Notes |
|---|---|---|
| `fig:supp:ica:decomposition` | none | Illustration, no plotting code (adapted from the MUniverse paper). |
| `fig:supp:ica:stationary` | none | Illustration, no plotting code. |
| `fig:supp:ica:muap_overlay` | none | Illustration, no plotting code. It shows one simulated MUAP across the joint-angle sweep; the pool and unit were not recorded. |
| `tab:supp:clusters`, `tab:supp:cluster_pose`, `tab:supp:ceiling` | none | Dataset construction: see the dataset bundle. |
| `tab:supp:dataset` and the construction stages | none | Dataset construction: see the dataset bundle. |

Dataset construction is not part of this code release. The benchmark,
<https://huggingface.co/datasets/emgbenchanon/muniverse-dynamic-v2>, ships the
pipeline as `reproducibility_bundle.zip`: the MUAP-cache generation,
crosstalk analysis, clean-cache selection and recording generation scripts,
the NeuroMotion patch and the pool configuration.

### Main-table analyses

| Result | Script | Notes |
|---|---|---|
| Methodology notes for Table 2 | `table1_*.sh` | The across-seed s.d. come from the table rows, and the recordings per tier (40 / 80 / 48) from the Table 1 block. For one seed's tier counts, run `yield_table.py` on that seed's CSV. The Adapt CSV holds all seeds and is reported only as mean ± s.d. |
| `tab:supp:detailed_yield` | `figures.sh` | `detailed_yield.csv`: mean per-recording yield at F1 ≥ 0.99 / 0.95 / 0.90 / 0.80 |
| `tab:supp:yield_per_rec` | `figures.sh` | `yield_per_rec.csv`: per tier, mean ± s.d. across recordings, and the median |
| `tab:supp:seed_std` | `figures.sh` | `seed_std.csv`: per tier, mean ± s.d. across seeds |
| `fig:supp:pairwise_overlap` | `figures.sh` | `overlap.py heatmap` writes `pairwise_overlap_heatmap.pdf` and `pairwise_overlap_summary.csv`. Inputs: HarmonICA (PE), the ICA reference, Chen, Kramberger with its default calibration, Yeung, the segmented Glaser (`run_glaser`) and Mendez Guerra 2024. `figures.sh` runs the two extra baselines itself. |
| `fig:supp:overlap_bars` | `figures.sh` | `overlap.py bars` writes `mu_overlap_stacked_bars.pdf`, `mu_overlap_by_bin.csv` and `mu_overlap_by_method.csv`. Same inputs as the heatmap, except that Glaser is the cyclostationary one. |
| `fig:supp:pareto_f1_ch320` and its dominance text | `figures.sh` | `pareto.py --axis f1`: `pareto_f1_ch320.pdf`, `pareto_f1_crossovers.csv`, `pareto_f1_dominance.txt` |
| `tab:supp:bootstrap` | `figures.sh` | `bootstrap.csv`: 95 % percentile interval, 5000 resamples. Kramberger is scored only on the recordings where it returns output. |
| `tab:supp:wilcoxon` | `figures.sh` | `wilcoxon.csv`: Δ, wins / losses / ties, `p` (two-sided) and `p_greater` (one-sided). The printed Glaser p-values are one-sided; the others are two-sided. |

### Implementation and baselines

| Result | Script | Notes |
|---|---|---|
| `tab:supp:impl`, settings | none | The operating point is `SurfaceConfig` (`src/harmonica/models/engine/configs.py`) plus the `run_signal_space` arguments the scripts pass. See also "Implementation notes" below. |
| `tab:supp:impl`, runtimes | `figures.sh` | `yield_table.py --runtime` per method and grid: wall time per (recording, seed) run. The ICA / cBSS reference run times preprocessing, FastICA and scoring. Adapt runs one single-threaded process per shard. |
| Compute budget | none | Arithmetic on the runtimes. |
| Hyperparameter selection (extension factor, compensator settings; Chen and Yeung settings) | `select_hparams.sh` | `PARTS=harmonica` runs per-unit HarmonICA on ch070: compensator rank, additive transform, epochs and warm-up at `R=4`, then `R` = 8 / 16 / 32 / 64. `PARTS=chen` and `PARTS=yeung` run Chen's `learning_rate` × `cov_forgetting` (3 × 3) and Yeung's `z_threshold` with the calibration `sil_th` (6 configurations), both grids. All three score ground-truth F1 on 24 test-split recordings (`experiments/lists/sweep24.txt`) with one seed. Chen and Yeung keep their wrappers' other defaults there (extension factor 10); Table 2 runs the chosen values at extension factor 8. `PARTS=val` re-checks HarmonICA (PE) on the validation split (40 recordings per grid, seeds 0–2): `R` = 4 / 8 / 16, warm-up 0, rank 1. Each table marks the setting the paper uses with `*`. |
| Woodbury arithmetic | none | Arithmetic. Implemented by `ConditionalLowRank` (`src/harmonica/models/engine/nets.py`). |
| Chen 2020, Yeung 2024 and Glaser 2018 settings | `table1_baselines.sh` | As in the Table 2 rows above. Chen's covariance update is an exponential moving average with `cov_forgetting`. |
| Adapt protocol and tuned values | `tune_adapt.sh`, `table1_adapt.sh` | The search is `tune_adapt_v1.py` on the validation split with one FastICA seed. It uses Optuna's multivariate TPE: 30 random start-up trials, 100 trials, seed 1909. Its objectives are Adapt's own whitening and separation losses on calibration units with CoV-ISI ≤ 0.3, with no ground truth, and the Pareto member with the lowest separation loss is kept. The paper's outputs are in `experiments/adapt_tuning/ch{70,320}_v1_pareto_cov_mvtpe30/`. `tune_adapt.sh` writes a new search to `$OUT/adapt_tuning/`. |
| Adapt: the twelve tuning variants | `adapt_variants.sh` | Reruns each search on the validation split (`STEPS=tune`) and evaluates each variant on the test split (`STEPS=eval`). It prints the yield per grid and summed over both grids. `experiments/adapt_tuning/` holds all 24 configurations, and `eval` reads them by default. |
| Adapt with default settings (v1.0.0 and v0 engines) | `table1_adapt.sh` | `bench_adapt.py --methods adapt,adapt_v1`: same harness, NeuroMotion configuration (v1.0.0) or the v0 defaults (learning rates 7e-3 / 3e-3) |
| Harness check: ICA / cBSS in the Adapt harness | `table1_adapt.sh`, `figures.sh` | `bench_adapt.py --methods ica_cbss` gives the yield row. The recall crossovers are the `ica_harness` rows of `pareto_recall_crossovers.csv`. |
| Other baselines in the Adapt author's harness | none | Not reproducible from this release. These are the method author's committed results from her own harness, and its baseline dispatchers are not included. |

The supplement's former Adapt calibration-window and learning-rate sweeps
(`tab:supp:adapt-calib-sweep`, `tab:supp:adapt-lr-sweep`) are withdrawn in
the camera-ready version and have no code here.

### Side information and ablations

| Result | Script | Notes |
|---|---|---|
| `tab:supp:pe_vs_angle` | `table1_harmonica.sh` with `PAPER_AS_PRINTED=1` | PE rows: ch070 pooled ("as printed"), ch320 seeds 0–4. θ rows: seeds 0–9. The PE side information is five sinusoids of time with periods 0.1, 0.2, 0.5, 1 and 2 s; θ is the joint angle. |
| `tab:supp:pe_vs_angle_pareto` and its dominance text | `figures.sh` with `PAPER_AS_PRINTED=1` | `pareto_recall_crossovers.csv`, rows HarmonICA, HarmonICA (θ), ICA and Mendez Guerra 2024; `pareto_recall_dominance.txt` checks both HarmonICA variants against every adaptive baseline |
| `tab:supp:headline_ablation` | `table16_ablations.sh`, `table1_harmonica.sh` | Reference row as for Table 3. Cold-start `w`: `warmstart_w=false`. Additive transform: `config_overrides={"compensation_model": "affine"}`, the element-wise `x·exp(log w(t)) + b(t)`. Coupled gradients: `couple_grads=true` (independence loss computed on the compensated source); its mean-F1 drop is the `mean F1` column against HarmonICA (PE). No peel-off: per-unit HarmonICA (`bench_sweep.py --method signal_space`, oracle pairing). Kurtosis loss: the filter-space refiner `refine_harmonica_style` with `comp_loss="kurtosis"`, the same runs as the kurtosis row of `tab:supp:loss`. |
| `tab:supp:loss` | `table17_loss_family.sh` | `refine_harmonica_style` (filter-space, per unit, oracle pairing, `R=8`) with `comp_loss` ∈ {`gmm`, `negentropy_g1`, `negentropy_g2`, `kurtosis`, `power_p`}, library defaults otherwise (150 epochs). Prints mean F1 and n at F1 ≥ 0.85 / 0.90 / 0.95 (the "total"). |
| `fig:supp:loss_rescue`, `fig:supp:loss_partial`, `fig:supp:loss_regression` (Figs 10–12) | none | Illustrations, no plotting code; see "Example figures" below. |
| `tab:supp:rank` | `table18_rank.sh` | `config_overrides={"compensation_rank": 5 or 7}`, `use_angle=true`. The r = 3 row is HarmonICA (θ) over seeds 0–4 from `table1_harmonica.sh`. |
| `tab:supp:pe_vs_angle_thresh` | `table1_harmonica.sh` | PE row: HarmonICA (PE), seeds 0–4. Its printed ch070 n at F1 ≥ 0.95 (476) is the pooled Table 2 value. θ row: HarmonICA (θ), seeds 0–4. Columns: mean F1, n at F1 ≥ 0.85 / 0.90 / 0.95. |
| Source examples (`supp:source_examples`, Figs 13–18) | none | Illustrations, no plotting code; see "Example figures" below. |
| `tab:supp:notation` | none | Settings only. The values are in `SurfaceConfig` and the scripts' arguments. |
| Nonlinear-ICA comparison (linear ICA, TCL, iVAE; added in the camera-ready version) | `table19_nonlinear_ica.sh` | `nonlinear_ica.py`. `--protocol angle` (linear ICA, TCL-θ, iVAE-θ) runs on 5 recordings per grid (`experiments/lists/nonlinear_ica_ch*.txt`), and `--protocol time` (TCL-t) on ch070 only. A unit counts as recovered at F1 > 0.5. The two protocols train TCL-θ differently and bracket the range the table gives for it on ch070 (script header). |

### Not reproducible from this release

- The real HD-EMG experiment: real data, not released. The recordings are
  human-subject data that cannot be redistributed, and the code for this
  experiment is not part of this release.

## Example figures

These figures have no plotting code. All of them use ch070 test recordings,
FastICA seed 1909, network seed 42 and `R=8`. Each unit starts from its
best-matching FastICA source, the pairing `bench_sweep.py` uses. Each figure
plots the joint angle and the sources with the ground-truth spikes. The F1
values below are the ones printed in the figure panels.

Loss figures: `refine_harmonica_style` once per `comp_loss` (`gmm`,
`negentropy_g1`, `negentropy_g2`, `kurtosis`, `power_p`), with `epochs=200`
(`tab:supp:loss` uses the default 150).

| Figure | Recording | GT unit | F1: ICA → GMM loss |
|---|---|---|---|
| `fig:supp:loss_rescue` (Fig. 10) | `SA10-lo_FCU_h_sub04_N036` | 31 | 0.643 → 1.000 |
| `fig:supp:loss_partial` (Fig. 11) | `TA10-lo_FDSI_sub02_N053` | 43 | 0.656 → 0.798 |
| `fig:supp:loss_regression` (Fig. 12) | `TA05-hi_ECU_sub01_N084` | 74 | 0.928 → 0.908 |

Source examples: per-unit HarmonICA (`refine_signal_space_comp_warmstart`)
with `epochs=200` and `sep_only_warmup=50`. Each unit is refined on the full
whitened recording, without peel-off.

| Figure (caption title) | Recording | GT unit | F1: ICA → HarmonICA |
|---|---|---|---|
| Fig. 13, Easy success | `SA05-hi_ECU_sub03_N055` | 48 | 1.000 → 1.000 |
| Fig. 14, ICA complete failure | `TA10-hi_FCU_u_sub03_N112` | 96 | 0.124 → 0.107 |
| Fig. 15, Stuck source rescued | `SA10-lo_FCU_h_sub04_N036` | 31 | 0.643 → 1.000 |
| Fig. 16, Partial success | `TA10-hi_FCU_u_sub03_N112` | 87 | 0.586 → 0.779 |
| Fig. 17, Noise amplification | `TA05-lo_FDSI_sub00_N045` | 39 | 0.833 → 0.585 |
| Fig. 18, zoom of Fig. 17 | `TA05-lo_FDSI_sub00_N045` | 39 | 5.3–8.3 s of Fig. 17 |

To recompute a unit's scores, run `bench_sweep.py` on a one-line recording
list. For the source examples:

```bash
echo SA10-lo_FCU_h_sub04_N036 > one.txt
python scripts/bench_sweep.py --data-dir "$DATA/ch070" --val-set one.txt \
    --method signal_space --kwargs '{"epochs":200,"sep_only_warmup":50}' \
    --R 8 --ica-seed 1909 --nn-seed 42 --tag example --output example.csv
```

The CSV has one row per ground-truth unit that a FastICA source matched (column `gt_mu`).

For the loss figures, use `--method harmonica_style --kwargs
'{"comp_loss":"gmm","epochs":200}'`.

## Implementation notes

Behaviour of the code that matters when you compare it with the text:

- **Band-pass.** Only `bench_adapt.py` band-passes. It filters every
  recording at 10–1000 Hz (zero-phase Butterworth) before calibration, for
  every method it runs, and switches Adapt's own filter off.
  `bench_baselines.py`, `bench_sweep.py` and `nonlinear_ica.py` use the
  recordings as provided, so HarmonICA and every other Table 2 row see
  unfiltered signals.
- **FastICA budget.** HarmonICA extracts up to `min(3 × n_channels, 108)` =
  108 candidates on both grids. The ICA / cBSS reference row
  (`bench_sweep.py`) uses `min(3 × n_gt, 108)` with the same seed, so for
  `n_gt` < 36 it extracts fewer candidates than HarmonICA starts from.
- **Duplicates.** `run_signal_space` trains every FastICA candidate for a
  fixed number of epochs and keeps them all. Scoring keeps each ground-truth
  unit's best match, so duplicate sources do not change the yield.
- **Engine operating point** (`SurfaceConfig`): rank-3 compensator, MLP
  (64, 64, 64) with dropout 0.2, independence-loss exponent 3.0, learning
  rates 1e-5 (separation) and 1e-4 (compensation), 1000 null samples. Spike
  timestamps come from a two-centroid soft k-means and are re-estimated
  every 5 compensation steps.
- **Matching.** A spike matches within ±2 ms after the best global shift in
  ±25 ms (`evaluate.py`).

## How close to expect

- **Table and figure scripts.** We ran them with `SKIP_RUNS=1` on the
  per-unit CSVs behind the paper (not part of this release), with
  `PAPER_AS_PRINTED=1` where the paper pooled seeds. Of the 782 printed cells they cover, all but 12 come out as
  printed:
  - Six are the HarmonICA (PE) totals in the reference rows of Table 3 and
    `tab:supp:headline_ablation`, and the PE n at F1 ≥ 0.95 of
    `tab:supp:pe_vs_angle_thresh`. They are explained under "Seed sets".
  - In `tab:supp:bootstrap`, the printed ch070 HarmonICA interval is
    [2.51, 3.14] against our [2.50, 3.17], and the ch320 Glaser upper bound
    is 0.61 against our 0.60.
  - In `tab:supp:wilcoxon`, the printed ch070 Δ is +1.47 for Chen (unrounded
    1.475), +2.23 for Kramberger (ours +2.25) and +2.10 for Yeung (ours
    +2.11).

  Another 10 cells sit exactly on a rounding tie (an unrounded value ending
  in 5). They match when rounded as Python prints them. The Pareto curves
  and overlap figures come out identical to the paper's.
- **Reruns on one machine.** HarmonICA, the ICA / cBSS reference, the
  filter-space refiner, TCL and Adapt seed every random draw they make, and a
  repeat run gave identical per-unit results. Chen, Yeung, both Glaser
  variants and Kramberger call scikit-learn's PCA without a seed, as in the
  paper's runs. Where scikit-learn picks its randomized solver, the result
  depends on NumPy's global random state. With scikit-learn 1.8 that covers
  the short calibration windows of Chen, Yeung and the segmented Glaser on
  both grids, and the whole-recording PCA of Kramberger and the
  cyclostationary Glaser on ch320. Their per-unit results can therefore
  change slightly from run to run. In a repeat run on one ch070 recording,
  Chen, Yeung and the segmented Glaser changed, and Kramberger and the
  cyclostationary Glaser did not. iVAE draws from PyTorch's global
  generator, which `nonlinear_ica.py` seeds (seed 42).
- **Across machines**, floating-point differences can change which FastICA
  filters are found. Expect agreement with the paper at the level of
  seed-to-seed variation, not to the last digit.
- The PCA step in `run_signal_space` now uses a fixed `random_state`. The
  paper's runs left it unset, so the projection differed slightly between
  runs.
- Adapt runs on the CPU. Repeated runs on one cluster gave identical
  per-unit results. A rerun on a second cluster reproduced 98.6–99.5 % of
  per-unit F1 values exactly and every per-seed total to within 3 units.
- TCL and iVAE vary strongly with how they are trained, which is why the
  table gives a range for TCL-θ on ch070.
