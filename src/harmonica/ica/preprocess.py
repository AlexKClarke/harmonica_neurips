"""Raw EMG band-pass filtering.

Operates on channels-first arrays ``(n_channels, n_samples)``, filtering
along the time axis with zero-phase :func:`scipy.signal.filtfilt`. The Adapt
harness (``scripts/bench_adapt.py``) band-passes at 10-1000 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_signals(
    emg_data: np.ndarray,
    fsamp: float,
    high_pass: float = 20,
    low_pass: float = 500,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the time axis.

    Args:
        emg_data: ``(n_channels, n_samples)`` — channels-first EMG.
        fsamp: Sampling frequency, in Hz.
        high_pass: Low-frequency cutoff (highpass edge), in Hz.
        low_pass: High-frequency cutoff (lowpass edge), in Hz.
        order: Butterworth order (per-edge; ``filtfilt`` doubles the effective
            order).

    Returns:
        Filtered EMG, same shape as the input.
    """
    b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
    return filtfilt(b, a, emg_data, axis=1)
