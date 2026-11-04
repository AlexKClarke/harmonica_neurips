# HarmonICA

Code for **HarmonICA: Quasi-Time-Invariant ICA for Motor Unit Decomposition
during Dynamic Contractions** (NeurIPS 2026).

HarmonICA decomposes high-density surface EMG recorded during movement into
motor-unit spike trains. A static ICA vector separates each motor unit after
a learned, low-rank, time-varying transform of the signal
(`A(t) = I + U(t) Vᵀ`), so the separator stays linear while the mixing drifts
with the joint angle. The vector is warm-started from FastICA. Training
alternates an independence step on the separation vector with a compensation
step on `A(t)`. Each recovered unit is removed through `A(t)⁻¹` before the next
one is trained (neural peel-off).

## Install

```bash
git clone <this repository> harmonica && cd harmonica
pip install -e .            # Python >= 3.10 and PyTorch; a CUDA GPU is recommended
pip install -e .[dev]       # adds pytest
pip install -e .[adapt]     # adds the dependencies of the Adapt (Mendez Guerra 2024) v1.0.0 engine
```

The `[adapt]` extra (optuna, plotly, seaborn, loguru) is needed only for the
Mendez Guerra 2024 rows: `run_adapt_v1`, `scripts/bench_adapt.py` and
`scripts/tune_adapt_v1.py`.

## Data

The benchmark (`muniverse-benchmark-dynamic`, simulated with MUniverse) is on
Hugging Face: <https://huggingface.co/datasets/emgbenchanon/muniverse-dynamic-v2>.
It has 168 test recordings for each of two electrode grids (70 and 320
channels), each 10 s at 2048 Hz.

- `recordings.tar.gz` extracts to `benchmark_v2/`, with `ch070/` and
  `ch320/`. Every script expects this directory (`DATA`).
- `recordings_val.tar.gz` extracts to `benchmark_v2_val/`: 40 recordings per
  grid from subjects disjoint from the test set (`VAL`). It is used only by
  the Adapt hyperparameter search and the hyperparameter-selection sweeps.
- `reproducibility_bundle.zip` holds the pipeline that built the dataset. This
  code release does not include it.

Each `.npz` holds `emg` (samples × channels), `fs`, `angle_profile` and one
`spike_mu_<k>` array of spike sample indices per ground-truth unit.

## Quick start

```python
import numpy as np
import harmonica

d = np.load("benchmark_v2/ch070/SA05-hi_ECRB_sub00_N071.npz", allow_pickle=True)
emg = d["emg"].T.astype(np.float32)          # channels-first: (n_channels, n_samples)
fs = int(d["fs"])
gt = {int(k.split("_")[-1]): d[k] for k in d.files if k.startswith("spike_mu_") and len(d[k])}

# Paper configuration: extension factor 8, 200 epochs, compensation from epoch 50.
spikes = harmonica.run_harmonica(emg, fs, R=8, epochs=200, sep_only_warmup=50)
scores = harmonica.evaluate_decomposition(spikes, gt, fs)
```

`run_harmonica` is an alias of `harmonica.baselines.run_signal_space`. By
default a sinusoidal positional encoding of time conditions the compensator
("HarmonICA (PE)"). For the angle-conditioned variant ("HarmonICA (θ)"),
pass the joint angle as the third argument together with `use_angle=True`.

## Reproducing the paper

[`REPRODUCING.md`](REPRODUCING.md) maps every table and figure of the paper and
the supplement to the script that produces it, with settings. It also covers
the seed sets and how close a rerun comes to the printed numbers. In short:

```bash
export DATA=/path/to/benchmark_v2 OUT=results
bash experiments/table1_harmonica.sh     # Table 1; HarmonICA (PE) and (θ) rows of Table 2 (GPU)
bash experiments/table1_baselines.sh     # ICA / cBSS, Chen, Kramberger, Yeung, Glaser rows
JOBS=16 bash experiments/table1_adapt.sh # Mendez Guerra 2024 (Adapt) rows (CPU)
bash experiments/figures.sh              # Fig. 3, overlap figures, per-recording and significance tables
```

By default every method runs seeds 0–4. `PAPER_AS_PRINTED=1` reproduces the
seed sets behind the printed numbers: HarmonICA (θ) over seeds 0–9, and the
ch070 HarmonICA (PE) row pooled with θ seeds 5–9. See
[`REPRODUCING.md`](REPRODUCING.md#seed-sets-the-default-and-paper_as_printed1).

## Repository layout

| Path | Contents |
|---|---|
| `src/harmonica/baselines/signal_space.py` | **HarmonICA** (`run_signal_space`) |
| `src/harmonica/models/engine/` | HarmonICA engine: compensator networks, the joint separation / compensation step, timestamping, neural peel-off ([README](src/harmonica/models/engine/README.md)) |
| `src/harmonica/models/harmonica_signal_space.py` | Per-unit HarmonICA without peel-off (no-peel-off ablation, source-example figures) |
| `src/harmonica/baselines/` | Baselines: Chen 2020, Kramberger 2021, Yeung 2024, Glaser 2018 (cyclostationary and segmented), Mendez Guerra 2024 (Adapt, v1.0.0 and v0 engines), TCL, iVAE |
| `src/harmonica/baselines/vendored_adapt_decomp{,_v1}/` | Vendored adapt_decomp (v0 snapshot and v1.0.0) |
| `src/harmonica/models/tcl.py`, `ivae.py` | TCL and iVAE models behind the nonlinear-ICA baselines |
| `src/harmonica/ica/` | FastICA / cBSS foundation: extension, whitening, fixed-point ICA, spike detection, peel-off, band-pass |
| `src/harmonica/nonstationary/` | Per-unit plumbing (`preprocess`, `run_ica`, `detect_best`) and the filter-space refiner `refine_harmonica_style` (loss-family ablation) |
| `src/harmonica/evaluate.py` | Spike matching and precision / recall / F1 |
| `scripts/` | Runners `bench_baselines.py` (multi-unit methods), `bench_sweep.py` (ICA reference and per-unit refiners), `bench_adapt.py` (Adapt rows), `nonlinear_ica.py` (TCL / iVAE); `tune_adapt_v1.py` (Adapt search); `yield_table.py` (table cells) |
| `scripts/figures/` | Pareto figures (`pareto.py`), overlap figures (`overlap.py`), per-recording and significance tables (`stats.py`) |
| `experiments/` | One script per table or group of tables, `figures.sh`, the hyperparameter sweeps, the recording lists and the Adapt search results (`adapt_tuning/`) |
| `tests/test_smoke.py` | Smoke tests |

[`MANIFEST.md`](MANIFEST.md) lists every file and the paper results it serves.

## Tests

```bash
pytest tests                                         # synthetic smoke tests (under a minute on a GPU)
HARMONICA_DATA=/path/to/benchmark_v2 pytest tests    # adds one benchmark recording
```

The Adapt test is skipped without the `[adapt]` extra.

## Troubleshooting

On some clusters, importing `torch` before `scipy` fails with a `GLIBCXX` error
because the system `libstdc++` is older than the one PyTorch was built against.
Import `harmonica`, which loads scipy first, before `torch`, or put your conda
environment's `lib/` on `LD_LIBRARY_PATH`.

## Citation

```bibtex
@inproceedings{clarke2026harmonica,
  title     = {HarmonICA: Quasi-Time-Invariant ICA for Motor Unit Decomposition during Dynamic Contractions},
  author    = {Clarke, Alexander Kenneth and Halatsis, Dimitrios and Grison, Agnese and
               Mendez Guerra, Irene and Ezaz-Nikpay, Noura and Mamidanna, Pranav and
               Ma, Shihan and Muceli, Silvia and Farina, Dario},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License and provenance

MIT (see [`LICENSE`](LICENSE)).

- `src/harmonica/baselines/vendored_adapt_decomp/` and
  `src/harmonica/baselines/vendored_adapt_decomp_v1/` are vendored copies of
  [imendezguerra/adapt_decomp](https://github.com/imendezguerra/adapt_decomp)
  by Irene Mendez Guerra (MIT). The first is commit `8ffeb66`. The second is
  release v1.0.0 at commit `a8f5373`: its `src/`, the NeuroMotion
  configuration, README, LICENSE and CITATION.cff.
- Irene Mendez Guerra contributed the Adapt v1 wrapper
  (`src/harmonica/baselines/adapt_v1.py`), `scripts/bench_adapt.py` and
  `scripts/tune_adapt_v1.py`.
- The engine in `src/harmonica/models/engine/` is adapted from a stand-alone
  HarmonICA implementation by Alexander Kenneth Clarke.
