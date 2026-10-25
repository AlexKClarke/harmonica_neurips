"""Raw EMG filtering — bandpass and notch.

Operates on channels-first arrays ``(n_channels, n_samples)``, filtering
along the time axis with zero-phase :func:`scipy.signal.filtfilt`.

Typical pipeline::

    emg = bandpass_signals(emg, fsamp=2048)          # 20-500 Hz
    emg = notch_signals(emg, fsamp=2048, nfreq=50)   # EU mains + harmonics

For US recordings set ``nfreq=60``.
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


def notch_signals(
    emg_data: np.ndarray,
    fsamp: float,
    nfreq: float = 50,
    dfreq: float = 1,
    order: int = 2,
    n_harmonics: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth notch at ``nfreq`` and its first harmonics.

    Notches at ``nfreq, 2*nfreq, ..., n_harmonics*nfreq``, each with
    stop-band half-width ``dfreq``. Applied sequentially.

    Args:
        emg_data: ``(n_channels, n_samples)`` — channels-first EMG.
        fsamp: Sampling frequency, in Hz.
        nfreq: Fundamental mains frequency, in Hz (50 for EU, 60 for US).
        dfreq: Stop-band half-width, in Hz.
        order: Butterworth order per notch.
        n_harmonics: Number of harmonics to notch (includes the fundamental
            when counting).

    Returns:
        Filtered EMG, same shape as the input.
    """
    harmonics = nfreq * np.arange(1, n_harmonics + 1)
    for h in harmonics:
        b, a = butter(order, [h - dfreq, h + dfreq], fs=fsamp, btype="bandstop")
        emg_data = filtfilt(b, a, emg_data, axis=1)
    return emg_data
