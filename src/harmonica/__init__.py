"""HarmonICA: quasi-time-invariant ICA for motor-unit decomposition.

Layout:

- :mod:`harmonica.ica` — the FastICA / cBSS foundation (extension,
  whitening, fixed-point ICA, spike detection, peel-off).
- :mod:`harmonica.baselines` — :func:`run_signal_space` (**HarmonICA**,
  the paper's method; also exported here as :func:`run_harmonica`) and the
  baseline methods it is compared against.
- :mod:`harmonica.models` — HarmonICA engine (``models.engine``),
  the per-unit variant ``refine_signal_space_comp_warmstart``, TCL / iVAE.
- :mod:`harmonica.nonstationary` — the filter-space refiner
  ``refine_harmonica_style`` (supplementary loss-family ablation), the
  per-unit benchmark plumbing (``preprocess``, ``run_ica``) and
  ``detect_best``.
- :mod:`harmonica.evaluate` — spike matching and F1 / precision / recall.

.. code-block:: python

    import harmonica                      # import before torch (see README)
    spikes = harmonica.run_harmonica(emg, fs, R=8, epochs=200, sep_only_warmup=50)
    scores = harmonica.evaluate_decomposition(spikes, gt_spikes, fs)
"""
from __future__ import annotations

__version__ = "1.0.0"

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
from .ica.preprocess import bandpass_signals
from .ica.whitening import whitening

# ── Per-unit refinement ─────────────────────────────────────────────
from .nonstationary.detect import detect_best
from .nonstationary.pipelines import preprocess, run_ica
from .nonstationary.refractory import refine_harmonica_style

# ── HarmonICA (the paper's method) ─────────────────────────────────
from .baselines.signal_space import run_signal_space

run_harmonica = run_signal_space

__all__ = [
    "bandpass_signals",
    "best_time_shift",
    "detect_best",
    "est_spike_times",
    "evaluate_decomposition",
    "match_spikes",
    "peel_off",
    "preprocess",
    "refine_harmonica_style",
    "run_harmonica",
    "run_ica",
    "run_ica_loop",
    "run_signal_space",
    "shift_aligned_match",
    "spike_triggered_average",
    "whitening",
]
