"""Per-unit HarmonICA: signal-space compensation, warm-started from ``w_ica``.

Runs the HarmonICA engine (:mod:`.engine`) on one warm-started unit, without
the rest of the multi-unit pipeline in
:func:`harmonica.baselines.signal_space.run_signal_space`:

- Projection pursuit, activity-based filter init, and peel-off are
  skipped.
- The separation vector is **warm-started from our ICA filter** ``w_ica``
  (the stand-alone engine starts from a random / activity-driven init).
- The patience / SIL / iteration gates are dropped — we just run the
  alternating ``joint_step`` loop for a fixed number of epochs.
- The compensation is applied **to the whitened signal** (``A(t) z_w``),
  not to the filter weights — this is the architectural twin of
  :func:`refine_harmonica_style` (which trains a filter-space
  compensation ``w_eff(t) = w_ica + dw(t)``).

Compare to:

- :func:`refine_iterative_refractory` — filter-space NN + refractory loss
- :func:`refine_harmonica_style`     — filter-space NN + trainable W
- this module                          — signal-space NN compensation
- :func:`harmonica.baselines.run_signal_space` — multi-unit HarmonICA
  with neural peel-off (the paper's method)

Without peel-off, nothing prevents the compensation NN from dragging the
source onto a co-firing neighbour; the multi-unit pipeline's peel-off is
what removes that failure mode. This module is used for the per-unit
ablations (no-peel-off row) and the source-extraction examples.
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
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Run the engine's ``joint_step`` alternating loop, warm-started from
    ``w_ica``, and return the compensated source signal.

    Args:
        z_w: ``(D, T)`` whitened signal (channels-first; matches the rest
            of our pipeline).
        w_ica: ``(D,)`` ICA separation direction (used as warm start for
            ``state.separation_weights``).
        t_norm: ``(T,)`` normalised time. *Unused*: the engine's
            ``State`` builds a sinusoidal positional encoding from
            ``positional_encoding_periods_s`` internally, so this
            function is always time-conditioned (no joint angle). Kept in the signature so this method has the same
            shape as :func:`refine_iterative_refractory` /
            :func:`refine_harmonica_style`.
        FS: Sampling frequency, in Hz.
        anchor_spikes: Ignored (no refractory / anchor loss). Kept in the
            signature for symmetry with the filter-space refiners.
        epochs: Number of alternating ``joint_step`` calls.
        sep_only_warmup: Number of initial epochs in which only the
            separation weight is updated (compensation delayed). The
            stand-alone engine uses 500; here ``0`` is fine because we
            warm-start from ``w_ica``.
        seed: Optional ``torch.manual_seed`` for reproducibility.
        config_overrides: Optional kwargs to override the default
            :class:`engine.SurfaceConfig`. Useful for ablations.
        return_info: If ``True`` return a ``(source, info)`` tuple where
            ``info`` carries epoch-wise diagnostics. Default ``False``
            (matches :func:`refine_harmonica_style`).

    Returns:
        ``(T,)`` compensated source signal in original time order, ready
        for :func:`harmonica.nonstationary.detect.detect_best`.
        If ``return_info`` is ``True``, returns a tuple
        ``(source, info)`` with ``info`` containing ``isi_history``,
        ``sil_history``, ``final_sil``, and ``epochs_run``.
    """
    import torch as _torch

    from .engine.configs import SurfaceConfig
    from .engine.decomposition import DecompositionModule
    from .engine.state import State

    if seed is not None:
        _torch.manual_seed(seed)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    D_, T = z_w.shape

    # ---- Build config — keep most engine defaults; disable gates -----
    config = SurfaceConfig(sampling_frequency=float(FS))
    config.acceptance_silhouette = 0.0     # accept anything (we're isolating)
    config.min_sil_for_compensation = 0.0
    config.min_peaks_in_source = 1
    config.verbose_mode = False
    config.log_every_n_steps = 0
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # The engine expects (T, D)
    signal_t = _torch.tensor(z_w.T, dtype=_torch.float32, device=device)

    # ---- Build State & module manually (skip preprocessing) ----------
    state = State(signal_t, config)
    state.init_exponent()
    # Override the random / activity-driven separation init with our w_ica
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

    info: dict = {"isi_history": [], "sil_history": [], "epochs_run": 0}

    # ---- Joint optimisation loop -------------------------------------
    for ep in range(epochs):
        use_comp = ep >= sep_only_warmup
        try:
            isi = model.joint_step(use_compensation=use_comp)
            info["isi_history"].append(int(isi))
        except Exception as e:
            info["error"] = f"joint_step error at ep {ep}: {type(e).__name__}: {e}"
            break
        info["epochs_run"] = ep + 1

        if (ep + 1) % 50 == 0:
            with _torch.no_grad():
                model.eval()
                _, sil = model.calculate_timestamps(calculate_silhouette=True)
                info["sil_history"].append(float(sil))
                model.train()

    # ---- Extract compensated source as (T,) numpy array --------------
    with _torch.no_grad():
        model.eval()
        comp_source = model.calculate_source(state.signal, compensated=True)
        comp_source_np = comp_source.detach().cpu().numpy().astype(np.float32)

    if return_info:
        # Compute final SIL once at the end for diagnostics
        try:
            with _torch.no_grad():
                model.eval()
                _, sil = model.calculate_timestamps(calculate_silhouette=True)
                info["final_sil"] = float(sil)
        except Exception:
            info["final_sil"] = float("nan")
        # LOCAL: expose the trained state so the separation filter and the
        # compensation network can be exported for release. Additive only.
        info["state"] = state
        return comp_source_np, info
    return comp_source_np
