# HarmonICA engine

PyTorch building blocks of HarmonICA: a separation vector trained jointly with
a time-varying, invertible compensator, spike timestamping, and peel-off
through the learned inverse. Adapted from a stand-alone package by Alexander
Kenneth Clarke; only the parts the paper's method uses are kept.

**How it is used.** `harmonica.baselines.run_signal_space` (HarmonICA) and the
per-unit `harmonica.models.harmonica_signal_space.refine_signal_space_comp_warmstart`
do their own preprocessing (Toeplitz extension, PCA, ZCA), build a `State` on
the whitened signal, warm-start each unit's separation vector from a FastICA
filter, train it for a fixed number of `joint_step` epochs (compensation
switched on after `sep_only_warmup`), timestamp the compensated source, and
(multi-unit only) call `peel_off` before the next unit. Configuration comes
from `SurfaceConfig` plus any `config_overrides`.

Data layout here is `(T, C)` (time-first), opposite to the `(C, T)` convention
used everywhere else in `harmonica`; the callers transpose at the boundary.

## Files

| File | Contents |
|---|---|
| `__init__.py` | Package surface: `Config`, `SurfaceConfig`, `DecompositionModule` |
| `configs.py` | Pydantic `Config` base and the `SurfaceConfig` preset (2.048 kHz surface EMG) |
| `decomposition.py` | `DecompositionModule`: `calculate_source`, `calculate_timestamps`, `joint_step`, `peel_off` |
| `state.py` | `State`: whitened signal, context signal, separation vector, compensator, optimisers |
| `nets.py` | Compensator networks `ConditionalLowRank` (default) and `ConditionalAffine` |
| `timestamping.py` | `peak_finder` (max-pool), `soft_k_means`, `signal_to_timestamps` (+ silhouette), `timestamps_to_windows` (STA windows) |

## Compensation

The compensator predicts a time-varying invertible map `A(t) = f_NN(context(t))`
applied to the whitened signal *before* the static separation vector `w`, so
the source is `wᵀ A(t) z(t)`. `context(t)` is the `auxiliary` argument of
`State`: either an observed signal (the joint angle, "HarmonICA (θ)") or, by
default, sinusoidal positional encodings of time at periods
`(0.1, 0.2, 0.5, 1.0, 2.0)` s (`State.get_auxiliary`, "HarmonICA (PE)").

Two architectures, selected by `Config.compensation_model`:

1. **`ConditionalLowRank`** (`"lowrank"`, default) — `A(t) = I + U(t) Vᵀ` with a
   learned shared `V ∈ ℝ^{C×r}` (`compensation_rank`, 3 by default) and
   per-sample `U(t) ∈ ℝ^{C×r}` predicted by an MLP. The inverse uses the
   Woodbury identity (closed-form solve for `r ≤ 3`).
2. **`ConditionalAffine`** (`"affine"`) — per-channel
   `x → x ⊙ exp(log_w(t)) + b(t)` from an MLP; invertible, but cannot mix
   channels. Used by the supplementary ablation.

Both cache their per-sample parameters (`init_cache` / `update_cache`), so the
compensated signal and its inverse can be evaluated on any slice of time.

## Training step

`joint_step` alternates two losses with two RMSprop optimisers:

- **Separation** (updates `w` only): z-score the *uncompensated* source,
  clamp at `clamp_max`, and minimise
  `L_sep = -mean(sign(s) · |nonlin_scale · s|^exponent)`. Gradient clipped at 10.
- **Compensation** (updates `A` only, `w` detached), once compensation is on:
  ```
  L_comp = sqrt( spike_std² + max(null_std², uncompensated_null_var) )
  ```
  over spike samples (timestamps of the compensated source, 50 % dropped at
  random) and `num_null_samples` null samples drawn away from the spikes.
  Timestamps are re-estimated every `compensation_resample_every` steps.
  Gradient clipped at 0.1.

## Neural peel-off

`peel_off(timestamps)` computes the spike-triggered average in the compensated
frame and, at each spike, maps it back through `A(t)⁻¹` (cached parameters,
sliced to the spike's window) before subtracting it from `State.signal`. The
template is thus removed through the learned time-varying map, not as a fixed
waveform.
