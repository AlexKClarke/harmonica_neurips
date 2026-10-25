"""Spike-count guardrail for the filter-space refiners.

:func:`proj_ratio_guardrail` — largely redundant after the
distance-weighted refractory loss. Kept as a safety net: when an NN-refined spike set shows count inflation but drops
its spatial alignment with the ICA filter, revert to ICA's spikes.

On the full ch070 benchmark (1625 MUs) the ``sr > 2.5`` default gives
``+0.006 mean Δ`` over iref, ``big_lo 154 → 109 (−29%)``, at the cost of
``big_win 194 → 172 (−11%)``.
"""
from __future__ import annotations

import numpy as np


def proj_ratio_guardrail(
    ica_spikes: np.ndarray,
    iref_spikes: np.ndarray,
    w_ica: np.ndarray,
    z_w: np.ndarray,
    pr_threshold: float = float("inf"),
    sr_threshold: float = 2.5,
) -> tuple[np.ndarray, dict]:
    """Revert to the ICA spike set when NN inflated and lost spatial alignment.

    Triggers when **both** features fire:

    - ``spike_ratio = n_iref / n_ica            > sr_threshold``
    - ``proj_ratio  = mean(|w_ica . z_w|_iref)
                    / mean(|w_ica . z_w|_ica)   < pr_threshold``

    Defaults (``pr_threshold = inf``, ``sr_threshold = 2.5``) effectively
    ignore the projection signal and trigger on spike-ratio alone — this
    is the best Pareto point from the full-benchmark sweep.

    History of prior thresholds:

    - ``pr<0.55`` alone:              0/58 real losses triggered — too conservative
    - ``pr<0.80 AND sr>2.0``:         +0.041 on failure pool, **−0.017 on full benchmark**
    - ``sr>2.5`` alone:               +0.006 on full benchmark — current default

    Args:
        ica_spikes: ICA-detected spike indices.
        iref_spikes: NN-refined spike indices.
        w_ica: ``(D,)`` unit-norm ICA filter.
        z_w: ``(D, T)`` whitened signal.
        pr_threshold: Projection-ratio trigger cutoff. Lower is more
            conservative.
        sr_threshold: Spike-ratio trigger cutoff. Higher is more
            conservative.

    Returns:
        ``(chosen_spikes, info)`` where ``info`` has ``proj_ratio``,
        ``spike_ratio`` and ``triggered``.
    """
    if len(ica_spikes) == 0 or len(iref_spikes) == 0:
        return iref_spikes, {
            "proj_ratio": 1.0, "spike_ratio": 1.0, "triggered": False,
        }
    anc_proj = np.abs(w_ica @ z_w[:, ica_spikes.astype(int)])
    iref_proj = np.abs(w_ica @ z_w[:, iref_spikes.astype(int)])
    proj_ratio = float(iref_proj.mean()) / (float(anc_proj.mean()) + 1e-8)
    spike_ratio = len(iref_spikes) / (len(ica_spikes) + 1e-8)

    triggered = (proj_ratio < pr_threshold) and (spike_ratio > sr_threshold)
    info = {
        "proj_ratio": proj_ratio,
        "spike_ratio": spike_ratio,
        "triggered": triggered,
    }
    if triggered:
        return ica_spikes.astype(int), info
    return iref_spikes.astype(int), info
