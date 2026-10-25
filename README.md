# HarmonICA

Code for **HarmonICA: Quasi-Time-Invariant ICA for Motor Unit Decomposition
during Dynamic Contractions** (NeurIPS 2026).

HarmonICA decomposes high-density surface EMG recorded during movement into
motor-unit spike trains. Each motor unit is separated by a static ICA vector
applied after a learned, low-rank, time-varying transform of the signal
(`A(t) = I + U(t) Vᵀ`), so the separator stays linear while the mixing drifts
with the joint angle. The vector is warm-started from FastICA. Training
alternates an independence step on the separation vector with a compensation
step on `A(t)`, and each recovered unit is removed through `A(t)⁻¹` before the
next one is trained (neural peel-off).

## Install

```bash
git clone <this repository> harmonica && cd harmonica
pip install -e .            # Python >= 3.10, PyTorch; a CUDA GPU is recommended
pip install -e .[dev]       # + pytest
```

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
default the compensator is conditioned on a sinusoidal positional encoding of
time ("HarmonICA (PE)"). Pass the joint angle as the third argument with
`use_angle=True` for the angle-conditioned variant ("HarmonICA (θ)").

## Data

The benchmark (`muniverse-benchmark-dynamic`, 168 recordings × 2 electrode grids,
simulated with MUniverse) is on Hugging Face:
<https://huggingface.co/datasets/emgbenchanon/muniverse-dynamic-v2>.
Download `recordings.tar.gz` and extract it. The scripts expect the resulting
`benchmark_v2/` directory, which holds `ch070/` and `ch320/`.

## Reproducing the paper

[`REPRODUCING.md`](REPRODUCING.md) maps every benchmark table to the script that
produces it. In short:

```bash
export DATA=/path/to/benchmark_v2
bash experiments/table1_harmonica.sh     # HarmonICA (PE) and (θ), both grids, 5 seeds
bash experiments/table1_baselines.sh     # Chen, Kramberger, Yeung, Glaser, ICA reference
```

## Repository layout

| Path | Contents |
|---|---|
| `src/harmonica/baselines/signal_space.py` | **HarmonICA** (`run_signal_space`) |
| `src/harmonica/models/engine/` | HarmonICA engine: compensator networks, joint separation / compensation step, timestamping, peel-off |
| `src/harmonica/models/harmonica_signal_space.py` | Per-unit HarmonICA (no peel-off), used for per-unit ablations |
| `src/harmonica/baselines/` | Baselines: Chen 2020, Kramberger 2021, Yeung 2024, Glaser 2018, Adapt (Mendez Guerra 2024, vendored), TCL, iVAE |
| `src/harmonica/ica/` | FastICA / cBSS foundation: extension, whitening, fixed-point ICA, spike detection, peel-off |
| `src/harmonica/nonstationary/` | Filter-space refiners (`refine_harmonica_style`, used in the loss-family ablation) |
| `src/harmonica/evaluate.py` | Spike matching, precision / recall / F1 |
| `scripts/` | `bench_baselines.py` (multi-unit methods), `bench_sweep.py` (per-unit refiners), `yield_table.py` (paper yield tables) |
| `experiments/` | One script per paper table, plus the recording lists |

## Tests

```bash
pytest tests                                  # synthetic smoke tests (~1 min on GPU)
HARMONICA_DATA=/path/to/benchmark_v2 pytest tests   # + one benchmark recording
```

## Troubleshooting

On some clusters, importing `torch` before `scipy` fails with a `GLIBCXX` error
because the system `libstdc++` is older than the one PyTorch was built against.
Import `harmonica` (which loads scipy first) before `torch`, or put your conda
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

## License

MIT (see [`LICENSE`](LICENSE)). `src/harmonica/baselines/vendored_adapt_decomp/`
is a vendored copy of
[imendezguerra/adapt_decomp](https://github.com/imendezguerra/adapt_decomp)
(MIT, commit `8ffeb66`).
