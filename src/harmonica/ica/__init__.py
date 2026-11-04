"""ICA tier — the foundation: preprocessing, whitening, ICA, detection, utils."""

from .algorithm import peel_off, run_ica_loop, spike_triggered_average
from .detect import est_spike_times
from .preprocess import bandpass_signals
from .utils import extension, gram_schmidt, remove_bad_sources, remove_duplicates
from .whitening import whitening

__all__ = [
    "bandpass_signals",
    "est_spike_times",
    "extension",
    "gram_schmidt",
    "peel_off",
    "remove_bad_sources",
    "remove_duplicates",
    "run_ica_loop",
    "spike_triggered_average",
    "whitening",
]
