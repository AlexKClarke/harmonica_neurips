"""Spike detection from a 1-D source signal.

The canonical detector is :func:`est_spike_times` — peak-find followed by
2-means clustering on peak amplitudes. It returns both the spike indices
and a silhouette-like (SIL) score used throughout the pipeline to gate
source acceptance (default ``sil_th = 0.8``).

More advanced detectors (template xcorr, two-gate, try-all) live in
``nonstationary.detect``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def est_spike_times(
    sig: np.ndarray,
    fsamp: float,
    cluster: str = "kmeans",
    a: float = 2,
    min_delay: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Detect spike indices via peak-find + 2-means clustering.

    Pipeline:

    1. Apply the asymmetric power law ``s -> sign(s) * |s|^a`` to enhance
       the contrast of positive peaks.
    2. Find peaks at least ``min_delay`` seconds apart.
    3. Cluster peak amplitudes into ``{spike, noise}`` with KMeans(k=2).
    4. Return the indices of the high-mean cluster plus a silhouette-like
       score ``(between - within) / max(between, within)``.

    Args:
        sig: Source signal, ``(n_samples,)``.
        fsamp: Sampling frequency, in Hz.
        cluster: Clustering method. Only ``'kmeans'`` is implemented.
        a: Exponent of the asymmetric power law used for contrast
            enhancement. Any positive real; the law is sign-preserving, so
            odd and fractional exponents behave like even ones.
        min_delay: Minimum peak separation, in seconds.

    Returns:
        ``(spike_indices, sil)``. Empty array and ``0.0`` if no peaks.
    """
    sig = np.asarray(sig)
    # Was ``np.sign(sig) * sig ** a``, which equals the documented
    # ``sign(s) * |s|^a`` only for EVEN ``a``. For odd ``a`` the two sign
    # factors cancel to ``|s|^a`` — full-wave rectification, so every negative
    # trough became a positive peak and ``find_peaks`` fired on it. For
    # fractional ``a`` a negative base gives NaN, wiping out half the signal.
    # Every caller uses the default ``a=2`` (even), so no existing result
    # changes; the bug only surfaced when sweeping the exponent.
    sig = np.sign(sig) * np.abs(sig) ** a

    if cluster != "kmeans":
        raise NotImplementedError(f"cluster={cluster!r} not supported")

    min_peak_dist = int(round(fsamp * min_delay))
    peaks, _ = find_peaks(sig, distance=min_peak_dist)

    if len(peaks) == 0:
        return np.array([], dtype=int), 0.0

    peak_vals = sig[peaks].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(peak_vals)
    centroids = kmeans.cluster_centers_.flatten()

    spike_cluster = int(np.argmax(centroids))
    est_spikes = peaks[labels == spike_cluster]

    # SIL: distances to own vs other centroid, squared.
    D = kmeans.transform(peak_vals)
    within = np.sum(D[labels == spike_cluster, spike_cluster] ** 2)
    between = np.sum(D[labels == spike_cluster, 1 - spike_cluster] ** 2)
    denom = max(within, between)
    sil = float((between - within) / denom) if denom > 0 else 0.0

    return est_spikes, sil
