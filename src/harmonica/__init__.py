"""HarmonICA: quasi-time-invariant ICA for motor-unit decomposition.

Layout:

- :mod:`harmonica.ica` — the FastICA / cBSS foundation (extension,
  whitening, fixed-point ICA, spike detection, peel-off).
- :mod:`harmonica.baselines` — :func:`run_signal_space` (**HarmonICA**,
  the paper's method; also exported here as :func:`run_harmonica`) and the
  baseline methods it is compared against.
- :mod:`harmonica.models` — HarmonICA engine (``models.engine``),
  the per-unit variant ``refine_signal_space_comp_warmstart``, TCL / iVAE.
- :mod:`harmonica.nonstationary` — filter-space refiners
  (``refine_harmonica_style`` and relatives), used in the loss-family
  ablation of the supplementary material.
- :mod:`harmonica.evaluate` — spike matching and F1 / precision / recall.

.. code-block:: python

    import harmonica                      # import before torch (see README)
    spikes = harmonica.run_harmonica(emg, fs, R=8, epochs=200, sep_only_warmup=50)
    scores = harmonica.evaluate_decomposition(spikes, gt_spikes, fs)
"""
from __future__ import annotations

__version__ = "0.1.0"

# ── evaluate (shared) ───────────────────────────────────────────────
from .evaluate import (
    best_time_shift,
    evaluate_decomposition,
    match_spikes,
    shift_aligned_match,
)

# ── ICA tier ────────────────────────────────────────────────────────
from .ica.algorithm import peel_off, run_ica_loop, spike_triggered_average
from .ica.detect import est_spike_times
from .ica.preprocess import bandpass_signals, notch_signals
from .ica.whitening import whitening

# ── Non-stationary tier (main refinement) ───────────────────────────
from .nonstationary.detect import detect_best, detect_spikes, template_spike_detect
from .nonstationary.harmonica import (
    gmm_cluster_mask,
    gmm_w1,
    iternn_refine,
    persample_refine,
)
from .nonstationary.pipelines import (
    nn_guided_peeloff,
    preprocess,
    refine_iternn,
    refine_peeloff_timenn,
    refine_timenn,
    run_all_methods,
    run_ica,
)
from .nonstationary.refractory import (
    refine_harmonica_style,
    refine_iterative_refractory,
)

# ── HarmonICA (the paper's method) ─────────────────────────────────
from .baselines.signal_space import run_signal_space

run_harmonica = run_signal_space

__all__ = [
    "bandpass_signals",
    "best_time_shift",
    "detect_best",
    "detect_spikes",
    "est_spike_times",
    "evaluate_decomposition",
    "gmm_cluster_mask",
    "gmm_w1",
    "iternn_refine",
    "match_spikes",
    "nn_guided_peeloff",
    "notch_signals",
    "peel_off",
    "persample_refine",
    "preprocess",
    "refine_harmonica_style",
    "refine_iternn",
    "refine_iterative_refractory",
    "refine_peeloff_timenn",
    "refine_timenn",
    "run_all_methods",
    "run_harmonica",
    "run_ica",
    "run_ica_loop",
    "run_signal_space",
    "shift_aligned_match",
    "spike_triggered_average",
    "template_spike_detect",
    "whitening",
]
