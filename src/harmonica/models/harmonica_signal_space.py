"""Per-unit HarmonICA: signal-space compensation, warm-started from ``w_ica``.

Runs the HarmonICA engine (:mod:`.engine`) on one unit, without the rest of
the multi-unit pipeline in
:func:`harmonica.baselines.signal_space.run_signal_space`:

- The separation vector is warm-started from a given FastICA filter
  ``w_ica``.
- The alternating ``joint_step`` loop runs for a fixed number of epochs.
- The compensation is applied **to the whitened signal** (``A(t) z_w``),
  not to the filter weights; :func:`harmonica.nonstationary.refine_harmonica_style`
  is the filter-space counterpart (``w_eff(t) = w + dw(t)``).
- There is no peel-off, so nothing prevents the compensator from drifting
  onto a co-firing neighbour; the multi-unit pipeline's neural peel-off
  removes that failure mode.

``scripts/bench_sweep.py --method signal_space`` refines each ground-truth
unit from its best-matching FastICA source with this function (the
no-peel-off ablation).
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# Public entry point
# =============================================================================


def refine_signal_space_comp_warmstart(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    anchor_spikes: np.ndarray | None = None,
    epochs: int = 200,
    sep_only_warmup: int = 0,
    seed: int | None = 42,
    config_overrides: dict | None = None,
) -> np.ndarray:
    """Run the engine's ``joint_step`` alternating loop, warm-started from
    ``w_ica``, and return the compensated source signal.

    Args:
        z_w: ``(D, T)`` whitened signal (channels-first).
        w_ica: ``(D,)`` ICA separation direction (used as warm start for
            ``state.separation_weights``).
        t_norm: ``(T,)`` normalised time. *Unused*: the engine's
            ``State`` builds a sinusoidal positional encoding from
            ``positional_encoding_periods_s`` internally, so this
            function is always time-conditioned (no joint angle). Kept so
            the signature matches :func:`refine_harmonica_style`.
        FS: Sampling frequency, in Hz.
        anchor_spikes: Ignored (no refractory / anchor loss). Kept in the
            signature for symmetry with the filter-space refiners.
        epochs: Number of alternating ``joint_step`` calls.
        sep_only_warmup: Number of initial epochs in which only the
            separation weight is updated (compensation delayed).
        seed: Optional ``torch.manual_seed`` for reproducibility.
        config_overrides: Optional kwargs to override the default
            :class:`engine.SurfaceConfig`.

    Returns:
        ``(T,)`` compensated source signal in original time order, ready
        for :func:`harmonica.nonstationary.detect.detect_best`.
    """
    import torch as _torch

    from .engine.configs import SurfaceConfig
    from .engine.decomposition import DecompositionModule
    from .engine.state import State

    if seed is not None:
        _torch.manual_seed(seed)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    D_, T = z_w.shape

    # ---- Build config ------------------------------------------------
    config = SurfaceConfig(sampling_frequency=float(FS))
    config.min_peaks_in_source = 1
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # The engine expects (T, D)
    signal_t = _torch.tensor(z_w.T, dtype=_torch.float32, device=device)

    # ---- Build State & module (z_w is already whitened) ---------------
    state = State(signal_t, config)
    state.init_exponent()
    # Warm-start the separation vector from w_ica
    state.separation_weights = _torch.nn.Parameter(
        _torch.tensor(w_ica, dtype=_torch.float32, device=device)
    )
    state.init_compensation_weights()
    state.init_optimisers()       # builds RMSprop on state.separation_weights

    model = DecompositionModule()
    model.config = config
    model.state = state
    model._comp_step_count = 0
    model._cached_spike_ts = None

    # ---- Joint optimisation loop -------------------------------------
    for ep in range(epochs):
        use_comp = ep >= sep_only_warmup
        try:
            model.joint_step(use_compensation=use_comp)
        except Exception:
            break

        # Timestamp the source every 50 epochs. The result is not used, but
        # signal_to_timestamps raises on a non-finite source, which aborts a
        # diverged unit here instead of returning a NaN source.
        if (ep + 1) % 50 == 0:
            with _torch.no_grad():
                model.eval()
                model.calculate_timestamps(calculate_silhouette=True)
                model.train()

    # ---- Extract compensated source as (T,) numpy array --------------
    with _torch.no_grad():
        model.eval()
        comp_source = model.calculate_source(state.signal, compensated=True)
        comp_source_np = comp_source.detach().cpu().numpy().astype(np.float32)
    return comp_source_np
