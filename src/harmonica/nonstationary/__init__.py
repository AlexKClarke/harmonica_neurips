"""Filter-space refiner (per unit, warm-started from an ICA filter).

:func:`refine_harmonica_style` trains a small network that perturbs the
separation filter over time (``w + dw(t)``) rather than transforming the
signal; it backs the supplementary stationarity-loss ablation. The paper's
method is the signal-space :func:`harmonica.baselines.run_signal_space`.
:func:`detect_best` turns a refined source into spikes.
"""

from .detect import detect_best
from .refractory import refine_harmonica_style

__all__ = [
    "detect_best",
    "refine_harmonica_style",
]
