# Reproducing the paper

Every benchmark number in the paper comes from per-recording CSVs written by
two runners:

- `scripts/bench_baselines.py` — multi-unit methods (HarmonICA and the
  baselines). One row per ground-truth unit: its best-matching recovered
  source and that source's precision / recall / F1.
- `scripts/bench_sweep.py` — per-unit refiners. Each ground-truth unit is
  refined from its best-matching FastICA source (an oracle pairing). The
  `*_ica` columns score the unrefined FastICA source.

`scripts/yield_table.py` turns one CSV per seed into the paper's yield cells:
units at F1 ≥ 0.95, binned by the recording's pool size (`n_gt` < 40, 40–69,
≥ 70), mean ± s.d. across seeds.

The scripts in `experiments/` run each table end to end and print its yield
rows. They loop over seeds and recordings serially. On an A40, HarmonICA takes
about 45 s per recording, so one seed of one grid is about 2 GPU-hours. To run
in parallel, set `SEEDS` and pass a single grid:

```bash
export DATA=/path/to/benchmark_v2 OUT=results
SEEDS=0 bash experiments/table1_harmonica.sh ch320
```

Seed index `i` means FastICA seed `1909 + i` and network seed `42 + i`. The
paper uses `i = 0..4`.

## Map

| Paper result | Script | Method and settings |
|---|---|---|
| Table 1 / supp. Table 14, HarmonICA (PE) | `table1_harmonica.sh` | `run_signal_space`, `R=8`, `epochs=200`, `sep_only_warmup=50`, `use_angle=false` |
| Table 1 / supp. Table 14, HarmonICA (θ) | `table1_harmonica.sh` | as above with `use_angle=true` (joint angle conditions the compensator) |
| Table 1, ICA / cBSS reference | `table1_baselines.sh` | `f1_ica` column of a `bench_sweep.py` run (FastICA at `R=8`; the extraction budget is `min(3 × n_gt, 108)`) |
| Table 1, Chen 2020 | `table1_baselines.sh` | `learning_rate=0.10`, `cov_forgetting=0.85`, `ext_fact=8` |
| Table 1, Kramberger 2021 | `table1_baselines.sh` | deterministic, one run; `R=8`, time-window calibration, `calib_sil_th=0.8` |
| Table 1, Yeung 2024 | `table1_baselines.sh` | `z_threshold=2.0`, `ext_fact=8` |
| Table 1, Glaser 2018 | `table1_baselines.sh` | `run_glaser_cyclostationary` (the paper's Pseudocode 1: cyclostationary CKC with PNR-gated tracking), `R=8` |
| Supp. §E.2, Adapt (Mendez Guerra 2024) | `table1_baselines.sh` | vendored reference code; `ext_fact=2`, `calib_s=5`, `wh_learning_rate=3e-3`, `sv_learning_rate=1e-3`. The wrapper now applies Adapt's own online preprocessing (causal 20–500 Hz bandpass, DC removal) before calibration, which the paper's runs did not, so this row can differ from §E.2 |
| Fig. 3, supp. Table 15 (yield × recall) | — | computed from the Table 1 CSVs (`r` column); plotting code not included |
| Supp. Table 16, cold-start `w_j` | `table16_ablations.sh` | `warmstart_w=false` |
| Supp. Table 16, additive transform | `table16_ablations.sh` | `config_overrides={"compensation_model": "affine"}` (element-wise `x·exp(log w(t)) + b(t)`) |
| Supp. Table 16, coupled gradients | `table16_ablations.sh` | `couple_grads=true` (independence loss computed on the compensated source) |
| Supp. Table 16, no peel-off | `table16_ablations.sh` | per-unit HarmonICA (`bench_sweep.py --method signal_space`) |
| Supp. Table 16, kurtosis loss; supp. Table 17 | `table16_ablations.sh`, `table17_loss_family.sh` | filter-space refiner `refine_harmonica_style`, `comp_loss ∈ {gmm, negentropy_g1, negentropy_g2, kurtosis, power_p}`, library defaults otherwise |
| Supp. Figs 10–12 | — | `refine_harmonica_style` with each `comp_loss`, `epochs=200`, seed 0 |
| Supp. Table 18, compensator rank | `table18_rank.sh` | `config_overrides={"compensation_rank": 5 or 7}`, angle-conditioned; `r=3` is HarmonICA (θ) |
| Supp. Table 19, TCL / iVAE | `table19_nonlinear_ica.sh` | runner defaults; scored at F1 > 0.5 |
| Supp. Figs 13–18 (source examples) | — | per-unit HarmonICA (`refine_signal_space_comp_warmstart`, `epochs=200`, `sep_only_warmup=50`, `R=8`, seed 0) |

The real-data experiment (§5.4, supp. Table 21) uses human recordings that we
cannot redistribute. Its code is not part of this release.

## How close to expect

Reruns agree with the paper at the level of seed-to-seed variation, not to the
last digit:

- Training runs on the GPU and is not bit-deterministic.
- The PCA step in `run_signal_space` now uses a fixed `random_state`. The
  paper's runs left it unset, so the projection differed slightly between runs.
- The TCL and iVAE baselines vary strongly across runs (the paper reports a
  range for TCL on ch070).
