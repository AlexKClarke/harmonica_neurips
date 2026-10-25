"""Filter-space refiners (per unit, warm-started from an ICA filter).

These train a small network that perturbs the separation filter over time
(``w + dw(t)``) rather than transforming the signal. The paper's method is
the signal-space :func:`harmonica.baselines.run_signal_space`; the refiners
here back the supplementary loss-family ablation
(:func:`refine_harmonica_style`) and the ICA reference row of the
benchmark tables (``f1_ica`` from :func:`refine_iterative_refractory` runs).
:func:`persample_refine` is the shared building block.
"""

from .detect import detect_best, detect_spikes, template_spike_detect
from .harmonica import (
    _PerSampleNet,
    gmm_cluster_mask,
    gmm_w1,
    iternn_refine,
    persample_refine,
)
from .refractory import refine_harmonica_style, refine_iterative_refractory

__all__ = [
    "detect_best",
    "detect_spikes",
    "gmm_cluster_mask",
    "gmm_w1",
    "iternn_refine",
    "persample_refine",
    "refine_harmonica_style",
    "refine_iterative_refractory",
    "template_spike_detect",
]
