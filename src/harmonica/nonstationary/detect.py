"""Spike detection for refined sources.

- :func:`template_spike_detect` — xcorr with an auto-built mean waveform.
- :func:`detect_spikes` — two-gate (KMeans on peaks ∩ absolute height),
  which keeps amplified noise peaks out of the spike cluster.
- :func:`detect_best` — runs k-means, template and two-gate detection on
  the positive peaks and keeps the detector that found the most spikes.
  Used for every per-unit refiner and the nonlinear-ICA baselines.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks
from sklearn.cluster import KMeans

from ..ica.detect import est_spike_times


# =============================================================================
# Template-based detection (xcorr with mean spike waveform)
# =============================================================================


def template_spike_detect(
    source: np.ndarray,
    fsamp: float,
    min_dist: int = 25,
) -> tuple[np.ndarray, float]:
    """Detect spikes by cross-correlation with an average spike template.

    Builds the template from the first 100 waveforms of initial k-means
    detections, cross-correlates it with the full source, then 2-means
    clusters the xcorr peaks to separate spike vs noise clusters.

    Args:
        source: ``(T,)`` source signal.
        fsamp: Sampling frequency (unused, kept for API consistency).
        min_dist: Minimum peak separation in the xcorr trace, in samples.

    Returns:
        ``(spikes, sil)``. Empty array / ``0.0`` on insufficient peaks.
    """
    peaks_r, _ = find_peaks(source, distance=20)
    if len(peaks_r) < 20:
        return np.array([]), 0.0

    km = KMeans(n_clusters=2, n_init=3, random_state=42)
    km.fit(source[peaks_r].reshape(-1, 1))
    sp_r = peaks_r[source[peaks_r] > km.cluster_centers_.mean()]
    if len(sp_r) < 10:
        return sp_r, 0.0

    # Build template from first 100 spike waveforms
    win = 25
    sp_v = sp_r[(sp_r >= win) & (sp_r < len(source) - win)]
    if len(sp_v) < 10:
        return sp_r, 0.0

    wf = np.array([source[int(s) - win : int(s) + win] for s in sp_v[:100]])
    tmpl = wf.mean(axis=0)
    tmpl /= np.linalg.norm(tmpl) + 1e-10

    xcorr = correlate(source, tmpl, mode="same")
    peaks, _ = find_peaks(xcorr, distance=min_dist)
    if len(peaks) < 10:
        return sp_r, 0.0

    km2 = KMeans(n_clusters=2, n_init=3, random_state=42)
    km2.fit(xcorr[peaks].reshape(-1, 1))
    final = peaks[xcorr[peaks] > km2.cluster_centers_.mean()]
    non_final = peaks[~np.isin(peaks, final)]

    if len(non_final) > 0:
        sil = (
            (xcorr[final].mean() - xcorr[non_final].mean())
            / (xcorr[final].std() + xcorr[non_final].std() + 1e-10)
        )
    else:
        sil = 0.0
    return final, float(sil)


# =============================================================================
# Two-gate detection (KMeans ∩ absolute height)
# =============================================================================


def detect_spikes(
    source: np.ndarray,
    FS: float,
    min_height: float = 3.0,
    min_delay_ms: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Two-gate spike detector: KMeans cluster AND height threshold.

    Gate 1: ``est_spike_times`` (k-means on asymmetric power-law peaks).
    Gate 2: ``peak > min_height * source.std()``.

    The intersection keeps refinement-amplified noise peaks out of the
    spike cluster.

    Args:
        source: ``(T,)`` source signal.
        FS: Sampling frequency, in Hz.
        min_height: Absolute amplitude threshold, in units of ``source.std()``.
        min_delay_ms: Minimum peak separation, in milliseconds.

    Returns:
        ``(spikes, sil)``.
    """
    sp_k, sil = est_spike_times(source, FS, min_delay=min_delay_ms / 1000.0)
    if len(sp_k) == 0:
        return sp_k, sil

    threshold = min_height * float(source.std())
    gated = sp_k[source[sp_k] > threshold]
    return gated, sil


# =============================================================================
# Try-all fallback detector
# =============================================================================


def detect_best(
    source: np.ndarray,
    FS: float,
    T: int,
    use_gated: bool = True,
    min_height: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Try every detector on the positive peaks, return the largest spike set.

    Only positive peaks are considered: the ICA refinement loop fixes the
    sign convention of each source.

    Detectors tried:

    - :func:`est_spike_times` (k-means on asymmetric power-law peaks)
    - :func:`template_spike_detect` (xcorr template matching)
    - :func:`detect_spikes` (two-gate) — when ``use_gated=True``

    Args:
        source: ``(T,)`` source signal.
        FS: Sampling frequency, in Hz.
        T: Signal length (used to clip out-of-range indices).
        use_gated: Whether to include the two-gate detector in the race.
        min_height: Height threshold for the two-gate detector.

    Returns:
        ``(spikes, sil)`` for the detector that found the most spikes.
    """
    best_sp: np.ndarray = np.array([], int)
    best_sil = 0.0
    best_n = 0

    # k-means
    sp, sil = est_spike_times(source, FS)
    sp = sp[(sp >= 0) & (sp < T)]
    if len(sp) > best_n:
        best_sp, best_sil, best_n = sp, sil, len(sp)

    # template xcorr
    sp_t, sil_t = template_spike_detect(source, FS)
    sp_t = sp_t[(sp_t >= 0) & (sp_t < T)]
    if len(sp_t) > best_n:
        best_sp, best_sil, best_n = sp_t, sil_t, len(sp_t)

    # two-gate
    if use_gated:
        sp_g, sil_g = detect_spikes(source, FS, min_height=min_height)
        sp_g = sp_g[(sp_g >= 0) & (sp_g < T)]
        if len(sp_g) > best_n:
            best_sp, best_sil, best_n = sp_g, sil_g, len(sp_g)

    return best_sp, best_sil
