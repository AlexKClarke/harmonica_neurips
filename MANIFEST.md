# Manifest

Every file in the release, what it does and which paper results it serves.
Main-paper items use the compiled numbering (Fig. 1–3, Table 1–3), and
supplementary items use their LaTeX labels.
[`REPRODUCING.md`](REPRODUCING.md) gives the commands and settings. Vendored
third-party trees are listed as one row each.

Abbreviations used below:

- **Table 2 runs**: every per-seed CSV behind Table 2. `figures.sh` reads all
  of them, so they also feed Fig. 3, `fig:supp:pareto_f1_ch320`, the two
  overlap figures, `tab:supp:detailed_yield`, `tab:supp:yield_per_rec`,
  `tab:supp:seed_std`, `tab:supp:bootstrap`, `tab:supp:wilcoxon` and
  `tab:supp:pe_vs_angle_pareto`.
- **HarmonICA results**: Table 2 (HarmonICA rows), Table 3,
  `tab:supp:headline_ablation`, `tab:supp:rank`, `tab:supp:pe_vs_angle` and
  `tab:supp:pe_vs_angle_thresh`.

## Top level

| File | Role | Paper results |
|---|---|---|
| `.gitignore` | Keeps caches, build outputs, benchmark data and run outputs out of the repository | none |
| `LICENSE` | MIT license, with the third-party notice for the vendored adapt_decomp | none |
| `MANIFEST.md` | This file | none |
| `README.md` | Overview, install, data, quick start, layout, citation, provenance | none |
| `REPRODUCING.md` | Maps each paper result to its script and settings, documents the seed sets, and says how close to expect | all |
| `pyproject.toml` | Package metadata (`harmonica-emg` 1.0.0), dependencies, the `[dev]` and `[adapt]` extras, and the package data of the vendored trees | all |

## `experiments/`: one entry point per table or group of results

| File | Role | Paper results |
|---|---|---|
| `experiments/_common.sh` | Shared settings (`DATA`, `OUT`, `SEEDS`, `THETA_SEEDS`, `PAPER_AS_PRINTED`, `JOBS`, `SKIP_RUNS`) and the runner wrappers `baselines`, `sweep`, `adapt`, `seed_csvs` and `table` | all scripted results |
| `experiments/table1_harmonica.sh` | Trains HarmonICA (PE) and (θ) on both grids and prints their rows and the ground-truth strata | Table 1; HarmonICA results |
| `experiments/table1_baselines.sh` | Runs Chen, Yeung, the cyclostationary Glaser, Kramberger and the ICA / cBSS reference, then prints their rows | Table 2 (these rows); Table 2 runs |
| `experiments/table1_adapt.sh` | Runs tuned Adapt, Adapt at default settings (v1.0.0 and v0) and ICA / cBSS in the Adapt harness, then prints their rows | Table 2 (Mendez Guerra 2024); the supplement's Adapt paragraph (default-settings yields, harness check); Table 2 runs |
| `experiments/figures.sh` | Builds the Pareto and overlap figures, the per-recording and significance tables and the runtimes from the Table 2 runs. Also runs the default-calibration Kramberger and the segmented Glaser that only the figures use | Fig. 3; Table 2 caption p-values; checklist runtimes; `fig:supp:pareto_f1_ch320`, `fig:supp:pairwise_overlap`, `fig:supp:overlap_bars`, `tab:supp:pe_vs_angle_pareto`, `tab:supp:detailed_yield`, `tab:supp:yield_per_rec`, `tab:supp:seed_std`, `tab:supp:bootstrap`, `tab:supp:wilcoxon`, `tab:supp:impl` runtimes |
| `experiments/table16_ablations.sh` | Ablations: cold-start `w`, additive transform, coupled gradients, no peel-off, kurtosis loss | Table 3, `tab:supp:headline_ablation` |
| `experiments/table17_loss_family.sh` | Stationarity-loss family on the filter-space refiner | `tab:supp:loss` |
| `experiments/table18_rank.sh` | Compensator rank 5 and 7, angle-conditioned | `tab:supp:rank` |
| `experiments/table19_nonlinear_ica.sh` | Linear ICA, TCL and iVAE (angle protocol on both grids, time protocol on ch070) | nonlinear-ICA comparison (camera-ready) |
| `experiments/tune_adapt.sh` | Reruns the Adapt hyperparameter search behind the Table 2 row, on the validation split | supplement, Adapt protocol and tuned values |
| `experiments/adapt_variants.sh` | Reruns the twelve Adapt tuning variants on the validation split and evaluates each on the test split | supplement, "one of twelve tuning variants" |
| `experiments/select_hparams.sh` | Hyperparameter-selection sweeps: HarmonICA extension factor and compensator settings, Chen and Yeung settings, and a validation-split check of HarmonICA | supplement, hyperparameter selection; checklist |
| `experiments/lists/ch070_all.txt` | The 168 ch070 test recordings | all ch070 benchmark results |
| `experiments/lists/ch320_all.txt` | The 168 ch320 test recordings (the same names as ch070) | all ch320 benchmark results |
| `experiments/lists/ch070_val.txt` | The 40 ch070 validation recordings, read by the validation part of `select_hparams.sh` | supplement, hyperparameter selection |
| `experiments/lists/ch320_val.txt` | The 40 ch320 validation recordings (the same names as ch070), read by the validation part of `select_hparams.sh` | supplement, hyperparameter selection |
| `experiments/lists/sweep24.txt` | The 24 test-split recordings of the HarmonICA, Chen and Yeung selection sweeps in `select_hparams.sh` (used on both grids) | supplement, hyperparameter selection and the Chen / Yeung settings |
| `experiments/lists/nonlinear_ica_ch070.txt` | The 5 ch070 recordings of the nonlinear-ICA comparison | nonlinear-ICA comparison |
| `experiments/lists/nonlinear_ica_ch320.txt` | The 5 ch320 recordings of the nonlinear-ICA comparison | nonlinear-ICA comparison |

### `experiments/adapt_tuning/`: Adapt search results

Each directory holds `best_params.json`, the selected configuration that
`bench_adapt.py --adapt-v1-opt-config` reads, and `all_trials.csv`, the trial
log of the search, which is kept as a record. `ch70` and `ch320` are the two
grids.

| Files | Role | Paper results |
|---|---|---|
| `experiments/adapt_tuning/ch{70,320}_v1_pareto_cov_mvtpe30/{best_params.json,all_trials.csv}` | The reported search: whitening and separation losses, CoV-ISI ≤ 0.3 pool, multivariate TPE with 30 start-up trials | Table 2 (Mendez Guerra 2024) and everything built on the Table 2 runs |
| `experiments/adapt_tuning/ch{70,320}_v1_pareto_paired/{best_params.json,all_trials.csv}` | Variant: both losses, ground-truth-paired pool | "one of twelve tuning variants" (`adapt_variants.sh`) |
| `experiments/adapt_tuning/ch{70,320}_v1_pareto_all/{best_params.json,all_trials.csv}` | Variant: both losses, all calibration units | same |
| `experiments/adapt_tuning/ch{70,320}_v1_pareto_sil/{best_params.json,all_trials.csv}` | Variant: both losses, SIL ≥ 0.9 pool | same |
| `experiments/adapt_tuning/ch{70,320}_v1_pareto_cov/{best_params.json,all_trials.csv}` | Variant: both losses, CoV-ISI ≤ 0.3 pool, default sampler | same |
| `experiments/adapt_tuning/ch{70,320}_v1_sv_paired/{best_params.json,all_trials.csv}` | Variant: separation loss, ground-truth-paired pool | same |
| `experiments/adapt_tuning/ch{70,320}_v1_sv_all/{best_params.json,all_trials.csv}` | Variant: separation loss, all calibration units | same |
| `experiments/adapt_tuning/ch{70,320}_v1_sv_sil/{best_params.json,all_trials.csv}` | Variant: separation loss, SIL ≥ 0.9 pool | same |
| `experiments/adapt_tuning/ch{70,320}_v1_sv_cov/{best_params.json,all_trials.csv}` | Variant: separation loss, CoV-ISI ≤ 0.3 pool, default sampler | same |
| `experiments/adapt_tuning/ch{70,320}_v1_sv_cov_mvtpe30/{best_params.json,all_trials.csv}` | Variant: separation loss, CoV-ISI ≤ 0.3 pool, multivariate TPE with 30 start-up trials | same |
| `experiments/adapt_tuning/ch{70,320}_v1_roa_paired/{best_params.json,all_trials.csv}` | Variant: rate of agreement with ground truth, ground-truth-paired pool | same |
| `experiments/adapt_tuning/ch{70,320}_v1_roa_all/{best_params.json,all_trials.csv}` | Variant: rate of agreement with ground truth, all calibration units | same |

## `scripts/`: runners and aggregation

| File | Role | Paper results |
|---|---|---|
| `scripts/bench_baselines.py` | Multi-unit runner: one method and its kwargs over a recording list, one row per matched ground-truth unit. Runs HarmonICA (`signal_space`), Chen, Yeung, both Glasers and Kramberger in the experiments; Adapt v0, TCL and iVAE are also registered | HarmonICA results; Table 2 runs; supplement, hyperparameter selection |
| `scripts/bench_sweep.py` | Per-unit runner: FastICA, then each ground-truth unit's best-matching source refined by `signal_space` or `harmonica_style`, or left unrefined (`--method ica`) | Table 2 (ICA / cBSS) and Table 2 runs; Table 3 and `tab:supp:headline_ablation` (no peel-off, kurtosis); `tab:supp:loss`; supplement, hyperparameter selection; scores of the example figures (Figs 10–18) |
| `scripts/bench_adapt.py` | Adapt runner: 10–1000 Hz band-pass, shared calibration on the first 1.5 s, then the tuned, default (v1.0.0, v0) or `ica_cbss` method; long-format CSV over all seeds | Table 2 (Mendez Guerra 2024) and Table 2 runs; supplement, Adapt paragraph (default settings, harness check, twelve variants) |
| `scripts/tune_adapt_v1.py` | Adapt hyperparameter search on the validation split with adapt_decomp v1.0.0's own optimisers; writes `best_params.json` and `all_trials.csv` | `experiments/adapt_tuning/`; supplement, Adapt protocol and tuned values |
| `scripts/nonlinear_ica.py` | Linear ICA, TCL and iVAE on a recording list; units recovered at F1 > 0.5 | nonlinear-ICA comparison |
| `scripts/yield_table.py` | Yield cells binned by pool size (mean ± s.d. across seeds), top-1000 F1, mean F1, extra thresholds, strata (`--strata`) and runtimes (`--runtime`) | Table 1; Table 2; Table 3; `tab:supp:headline_ablation`, `tab:supp:loss`, `tab:supp:rank`, `tab:supp:pe_vs_angle`, `tab:supp:pe_vs_angle_thresh`; `tab:supp:impl` runtimes; supplement, Adapt paragraph |
| `scripts/figures/_inputs.py` | Reads the per-seed and long-format result CSVs into one frame for the three scripts below | as below |
| `scripts/figures/pareto.py` | Yield × recall and yield × F1 Pareto curves, threshold crossovers, dominance margins | Fig. 3 and its in-text ratios; `fig:supp:pareto_f1_ch320`; `tab:supp:pe_vs_angle_pareto`; Adapt harness check (recall) |
| `scripts/figures/overlap.py` | Which units the methods share at F1 ≥ 0.95: pairwise heatmap and stacked bars against ICA | `fig:supp:pairwise_overlap`, `fig:supp:overlap_bars`; the in-text discovery claim |
| `scripts/figures/stats.py` | Per-recording yields, the tables built on them, bootstrap intervals, paired Wilcoxon tests | `tab:supp:detailed_yield`, `tab:supp:yield_per_rec`, `tab:supp:seed_std`, `tab:supp:bootstrap`, `tab:supp:wilcoxon`; Table 2 caption p-values |

## `src/harmonica/`: the package

| File | Role | Paper results |
|---|---|---|
| `src/harmonica/__init__.py` | Public API; `run_harmonica` is an alias of `run_signal_space` | none (API) |
| `src/harmonica/evaluate.py` | Spike matching (±2 ms after the best shift in ±25 ms), per-source precision / recall / F1, best F1 per ground-truth unit | every benchmark number |

### `src/harmonica/ica/`: FastICA / cBSS foundation

| File | Role | Paper results |
|---|---|---|
| `src/harmonica/ica/__init__.py` | Exports of the ICA layer | none (API) |
| `src/harmonica/ica/algorithm.py` | `run_ica_loop` (fixed-point FastICA with SIL gate, optional CoV refinement and peel-off), `peel_off`, `spike_triggered_average` | HarmonICA's warm start; the ICA / cBSS reference; the calibration of Chen, Yeung, both Glasers and Adapt; TCL's feature ICA |
| `src/harmonica/ica/detect.py` | `est_spike_times`: peak finding, 2-means on peak heights, SIL score | every method |
| `src/harmonica/ica/preprocess.py` | `bandpass_signals`, the zero-phase band-pass of the Adapt harness | Mendez Guerra 2024 rows and the Adapt search |
| `src/harmonica/ica/utils.py` | Toeplitz `extension`, `gram_schmidt`, duplicate and bad-source removal | every method |
| `src/harmonica/ica/whitening.py` | ZCA / PCA / Cholesky whitening | every method |

### `src/harmonica/baselines/`: HarmonICA and the compared methods

| File | Role | Paper results |
|---|---|---|
| `src/harmonica/baselines/__init__.py` | `run_<method>` registry; imports `run_adapt_v1` only on first use | none (API) |
| `src/harmonica/baselines/signal_space.py` | **HarmonICA** (`run_signal_space`): FastICA warm start, low-rank compensator, alternating training, neural peel-off; ablation switches `warmstart_w`, `couple_grads` and `config_overrides` | HarmonICA results; Table 2 runs; supplement, hyperparameter selection |
| `src/harmonica/baselines/chen.py` | Chen et al. 2020, adaptive CKC | Table 2 (Chen) and Table 2 runs; supplement, hyperparameter selection |
| `src/harmonica/baselines/yeung.py` | Yeung et al. 2024, directional-forgetting RLS | Table 2 (Yeung) and Table 2 runs; supplement, hyperparameter selection |
| `src/harmonica/baselines/kramberger.py` | Kramberger & Holobar 2021, angle-driven filter tracking | Table 2 (Kramberger) and Table 2 runs; its default calibration feeds the Pareto curves and overlap figures |
| `src/harmonica/baselines/glaser_cyclostationary.py` | Glaser & Holobar 2018, cyclostationary CKC with PNR-gated tracking | Table 2 (Glaser) and Table 2 runs |
| `src/harmonica/baselines/glaser.py` | Glaser & Holobar 2018, segmented cBSS with stitching | `fig:supp:pairwise_overlap` |
| `src/harmonica/baselines/adapt_v1.py` | Mendez Guerra 2024 on the adapt_decomp v1.0.0 engine, calibrated with the shared ICA stack | Table 2 (Mendez Guerra 2024) and Table 2 runs; supplement, Adapt paragraph |
| `src/harmonica/baselines/adapt.py` | Mendez Guerra 2024 on the v0 engine; also the calibration helpers that `adapt_v1.py` and `tune_adapt_v1.py` reuse | supplement, Adapt paragraph (v0 default-settings yield); Mendez Guerra 2024 rows (shared helpers) |
| `src/harmonica/baselines/tcl.py` | `run_tcl`: TCL on the shared whitened input, angle bins (TCL-θ) or time segments (TCL-t) | nonlinear-ICA comparison |
| `src/harmonica/baselines/ivae.py` | `run_ivae`: iVAE on the shared whitened input with the angle as auxiliary variable | nonlinear-ICA comparison |
| `src/harmonica/baselines/vendored_adapt_decomp/` (7 files) | Vendored [imendezguerra/adapt_decomp](https://github.com/imendezguerra/adapt_decomp) at commit `8ffeb66` (v0): `adaptation.py`, `config.py`, `data_structures.py`, `io.py`, `preprocessing.py`, `__init__.py` (provenance note), `LICENSE_adapt_decomp`. Imports made relative | supplement, Adapt paragraph (v0 default-settings yield) |
| `src/harmonica/baselines/vendored_adapt_decomp_v1/` (30 files) | Vendored adapt_decomp release v1.0.0 at commit `a8f5373`: `src/adapt_decomp/` (adaptation, cbss, preprocessing, spikes, utils), `configs/adapt_configs/default_neuromotion.yaml`, `README.md`, `LICENSE`, `CITATION.cff` | Table 2 (Mendez Guerra 2024) and Table 2 runs; supplement, Adapt paragraph and the Adapt search |

### `src/harmonica/models/`: HarmonICA engine and nonlinear-ICA models

| File | Role | Paper results |
|---|---|---|
| `src/harmonica/models/__init__.py` | Exports `refine_signal_space_comp_warmstart` | none (API) |
| `src/harmonica/models/harmonica_signal_space.py` | Per-unit HarmonICA: one warm-started filter, the engine's alternating loop, no peel-off | Table 3 and `tab:supp:headline_ablation` (no peel-off); source-example figures (Figs 13–18); supplement, hyperparameter selection |
| `src/harmonica/models/tcl.py` | TCL model: segment classifier, feature extraction, ICA on the features | nonlinear-ICA comparison |
| `src/harmonica/models/ivae.py` | iVAE model, training and source extraction | nonlinear-ICA comparison |
| `src/harmonica/models/engine/__init__.py` | Exports `Config`, `SurfaceConfig` and `DecompositionModule` | none (API) |
| `src/harmonica/models/engine/README.md` | How the engine is used, its files, the two compensators, the training step and neural peel-off | none (documentation) |
| `src/harmonica/models/engine/configs.py` | `Config` base and the `SurfaceConfig` operating point (rank, MLP, learning rates, exponent, PE periods) | HarmonICA results; `tab:supp:impl`, `tab:supp:notation` |
| `src/harmonica/models/engine/decomposition.py` | `DecompositionModule`: source, timestamps, the alternating `joint_step`, neural `peel_off` | HarmonICA results; Algorithm 1 |
| `src/harmonica/models/engine/nets.py` | Compensators: `ConditionalLowRank` (`I + U(t)Vᵀ`, Woodbury inverse) and `ConditionalAffine` (additive-transform ablation) | HarmonICA results; Woodbury arithmetic; additive-transform row |
| `src/harmonica/models/engine/state.py` | `State`: whitened signal, context signal (positional encoding or angle), weights, optimisers | HarmonICA results |
| `src/harmonica/models/engine/timestamping.py` | Peak finding, soft k-means spike timestamping, spike-triggered windows | HarmonICA results |

### `src/harmonica/nonstationary/`: per-unit plumbing and the filter-space refiner

| File | Role | Paper results |
|---|---|---|
| `src/harmonica/nonstationary/__init__.py` | Exports `detect_best` and `refine_harmonica_style` | none (API) |
| `src/harmonica/nonstationary/pipelines.py` | `preprocess` (extension, PCA, ZCA), `run_ica` (FastICA, budget `min(3 × n_gt, 108)`), `_prf` (per-unit scores) | Table 2 (ICA / cBSS) and Table 2 runs; `tab:supp:loss`; Table 3 (no peel-off, kurtosis); nonlinear-ICA comparison |
| `src/harmonica/nonstationary/detect.py` | `detect_best`: k-means, template and two-gate detection, keeps the detector with the most spikes | every per-unit result; nonlinear-ICA comparison |
| `src/harmonica/nonstationary/refractory.py` | `refine_harmonica_style`, the filter-space refiner (`w + dw(t)`), with its stationarity-loss family and refractory term | `tab:supp:loss`; kurtosis row of Table 3 and `tab:supp:headline_ablation`; loss figures (Figs 10–12) |
| `src/harmonica/nonstationary/harmonica.py` | `gmm_cluster_mask` and the per-sample network that `refine_harmonica_style` trains | as `refractory.py` |

## `tests/`

| File | Role | Paper results |
|---|---|---|
| `tests/test_smoke.py` | Checks that the API imports and that the detector is sign-preserving. HarmonICA and Adapt v1.0.0 (from the vendored tree) must each recover at least two of three units of a synthetic convolutive mixture at F1 ≥ 0.9. Runs the paper configuration on one benchmark recording when `HARMONICA_DATA` is set | none (tests) |
