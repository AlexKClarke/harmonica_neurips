# HarmonICA engine

PyTorch implementation of the HarmonICA building blocks: gradient-based
separation with a jointly trained, time-varying compensation network, spike
timestamping, and peel-off through the learned inverse. Originally written by
Alexander Kenneth Clarke as a stand-alone package.

The stand-alone pipeline (`DecompositionModule.run`) is:

```
raw signal (T, C)
  → bandpass
  → extend (factor=100)
  → whiten (ZCA)
  → projection-pursuit loop:
        gradient ICA  +  jointly-trained NN compensation A(θ, t)
        → timestamp via soft-k-means on peaks + silhouette
        → accept / reject
        → angle-dependent STA peel-off (via the compensation inverse)
        → repeat
  → list of timestamp tensors (one per MU)
```

**How the paper uses it.** `harmonica.baselines.run_signal_space` does not call
`run`. It does its own preprocessing (extension, PCA, ZCA), warm-starts each
unit's separation vector from a FastICA filter, disables the silhouette /
patience gates, trains each unit for a fixed number of `joint_step` epochs
(compensation switched on after `sep_only_warmup`), and calls `peel_off`
between units. Configuration comes from `SurfaceConfig` plus any
`config_overrides`.

Data layout here is `(T, C)` (time-first), opposite to the `(C, T)` convention
used everywhere else in `harmonica`; the callers transpose at the boundary.

## Files

| File | Contents |
|---|---|
| `__init__.py` | Package surface: `Config`, `IntramuscularConfig`, `SurfaceConfig`, `DecompositionModule` |
| `configs.py` | Pydantic config base + two presets (intramuscular @ 10.24 kHz, surface @ 2.048 kHz) |
| `decomposition.py` | Main `DecompositionModule` — joint `joint_step`, `run_decomposition`, `peel_off`, `run` (projection-pursuit loop) |
| `state.py` | `State` container: preprocessed signal, auxiliary (context) signal, activity indices, separation/compensation weights, optimisers |
| `nets.py` | Three compensation network architectures (see below) |
| `preprocessing.py` | `high_pass_filter`, `low_pass_filter`, `extend` (Toeplitz), `whiten` (chol/zca/pca/zca_cor/pca_cor) |
| `timestamping.py` | `peak_finder` (maxpool), `soft_k_means`, `signal_to_timestamps`, `timestamps_to_windows` (STA extractor), `find_quality_metric` (shift-tolerant accuracy/precision/recall), `silhouette_measure` |
| `plotting.py` | `plot_sources` — live training plot (save_dir) |
| `logger.py` | Run-directory / JSON log writer |

## Non-stationary compensation

The compensation network predicts a time-varying invertible map `A(t) =
f_NN(context(t))` applied to the whitened signal *before* the static separation
vector `w` — so the effective filter is `w ∘ A(t)`. `context(t)` comes from the
`auxiliary` argument, which is **either**:

- an **external observed signal** (e.g. joint angle), passed in via `run(..., auxiliary=…)`, **or**
- auto-generated sinusoidal positional encodings at periods `(0.1, 0.2, 0.5, 1.0, 2.0)` seconds (`State.get_auxiliary`).

This means it can run **blind** (sinusoidal context = a learned time-NN) or
**semi-blind** (observed context = HarmonICA-style angle conditioning).

### Three compensation architectures (`nets.py`)

Switchable via `Config.compensation_model`:

1. **`ConditionalAffine`** (`"affine"`) — per-sample `log_weight(t)` and
   `bias(t)` from an MLP: `x → x * exp(log_w(t)) + b(t)`. Element-wise in
   channel — *cannot rotate the filter* (only rescales/shifts).

2. **`ConditionalGivensMatrix`** (`"givens"`) — `Affine` + a stack of learned
   Givens rotations on random channel pairs, with angles predicted by a small
   MLP. Invertible, cheap, **can rotate** in the selected 2-planes.

3. **`ConditionalLowRank`** (`"lowrank"`, **DEFAULT**) — `A(t) = I + U(t) Vᵀ`
   with a learned shared `V ∈ ℝ^{C×r}` (rank 3 default) and per-sample
   `U(t) ∈ ℝ^{C×r}` predicted by an MLP. Inverse via **Woodbury identity**
   (closed-form solve for r≤3). Full per-sample correction of rank `r` — the
   most expressive of the three, and the default.

All three support `init_cache` / `update_cache` so compensation params are
computed once per update and indexed with a slice for peel-off.

### Training — two losses, two optimizers, alternating

`joint_step` in `decomposition.py`:

- **Separation loss** (ICA): normalize source to z-score, clamp to `clamp_max=30`,
  power-law non-Gaussianity `L_sep = -mean(sign(s) * |nonlin_scale * s|^exponent)`.
  Updates only `separation_weights`. Gradient clipped at 10.
- **Compensation loss** (spike/null contrast):
  ```
  L_comp = sqrt( spike_std² + max(null_std², uncompensated_null_var) )
  ```
  where spike/null vectors are sampled via timestamps from the current source
  (+ null = rejection sampling outside timestamps). **Spike dropout 0.5**,
  **1000 null samples**. Only updates compensation weights. Gradient clipped at 0.1.

Separate RMSprop optimizers. Compensation is **delayed** by
`compensation_delay=100` steps — the ICA filter settles first, then
compensation joins. `max_decomp_steps=1000` per source, early stop on patience
(100 steps no ISI improvement) or silhouette below 0.5.

### Stability guards

- **Baseline drift monitor**: measures variance of segment means. If the
  compensated source drifts >3× reference drift (recorded at
  `compensation_delay`), compensation is stopped → prevents collapse.
- **Silhouette gate**: if `sil < 0.5` at `compensation_delay`, end early
  (compensation won't rescue a hopeless source).
- **Compensation cache resampling**: recompute spike/null peaks every N steps
  (`compensation_resample_every=5`) — spikes/nulls are stale between resamples.

### Projection pursuit + angle-dependent peel-off

After each accepted source, `peel_off()` in `decomposition.py`:

1. Compute the STA in **compensated space** (`windows.mean(0)`).
2. For each spike time `s`, use
   `compensation_weights.inverse(sta, auxiliary, use_cache=True, indices=slice(start, end))`
   to **transform the STA back through the time-varying `A(t)⁻¹`** and subtract
   it from the raw (whitened) signal.

The STA is subtracted through the learned time-varying map, not as a fixed
template.

Acceptance criteria: `silhouette ≥ 0.9`, `≥10` spikes, and **not a repeat** —
repeats detected via `find_quality_metric` with shift tolerance
(`accuracy_max_shift_ms=10`); if `>90% accuracy` overlap with a prior source,
peel-off is triggered anyway to reduce the signal.
