"""ICA-tier utilities — Toeplitz extension, Gram-Schmidt, duplicate and
bad-source removal (used by ``run_ica_loop`` and the baselines).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from ..evaluate import get_bin_spikes, match_spike_trains, max_xcorr


# =============================================================================
# Toeplitz extension
# =============================================================================


def extension(Y: np.ndarray, R: int) -> np.ndarray:
    """Extend a channels-first signal by ``R`` delayed copies (Toeplitz).

    Produces a ``(n_channels * R, n_samples)`` matrix where each original
    channel is stacked with its ``R - 1`` delayed copies.

    Args:
        Y: ``(n_channels, n_samples)`` — channels-first signal.
        R: Extension factor (number of delayed copies, including lag 0).

    Returns:
        Extended signal, ``(n_channels * R, n_samples)``.
    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples))
    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        eY[i * R : (i + 1) * R, :] = toeplitz(col, row)
    return eY


# =============================================================================
# Deflation: Gram-Schmidt orthogonalization
# =============================================================================


def gram_schmidt(w: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stabilized Gram-Schmidt — orthogonalize ``w`` against columns of ``B``.

    Zero columns of ``B`` are skipped.

    Args:
        w: Vector to be orthogonalized, ``(n,)``.
        B: Basis whose columns ``w`` should become orthogonal to, ``(n, k)``.

    Returns:
        Orthogonalized vector, ``(n,)``.
    """
    w = np.asarray(w, dtype=float)
    B = np.asarray(B, dtype=float)

    non_zero_cols = ~np.all(B == 0, axis=0)
    B = B[:, non_zero_cols]

    u = w.copy()
    for i in range(B.shape[1]):
        a = B[:, i]
        projection = (np.dot(u, a) / np.dot(a, a)) * a
        u = u - projection
    return u


# =============================================================================
# Post-ICA cleanup: dedup + bad-source removal
# =============================================================================


def remove_duplicates(
    sources: np.ndarray,
    spikes: dict[int, np.ndarray],
    sil: np.ndarray,
    mu_filters: np.ndarray,
    fsamp: float,
    max_shift: float = 0.1,
    tol: float = 0.001,
    threshold: float = 0.3,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Merge duplicate sources by spike-train matching.

    For every ordered pair ``(i, j)``, computes the xcorr-shift of their
    binary spike trains then counts common spikes. If the common fraction
    exceeds ``threshold``, ``j`` is relabelled as a duplicate of ``i``.
    Within each resulting cluster the source with the highest SIL is
    retained.

    Args:
        sources: ``(n_mu, n_samples)`` source signals.
        spikes: ``{i: spike_indices}`` — spike indices per source.
        sil: ``(n_mu,)`` SIL scores.
        mu_filters: ``(n_features, n_mu)`` extraction filters.
        fsamp: Sampling frequency, in Hz.
        max_shift: Maximum xcorr lag considered, in seconds.
        tol: Spike-matching tolerance, in seconds.
        threshold: Minimum common-spike fraction to merge.

    Returns:
        ``(new_sources, new_spikes, new_sil, new_filters)`` with duplicates
        collapsed to the highest-SIL representative.
    """
    n_source = sources.shape[0]
    new_labels = np.arange(n_source)

    for i in range(n_source):
        if new_labels[i] < i:  # already absorbed into an earlier cluster
            continue
        st1 = get_bin_spikes(spikes[i], sources.shape[1])
        for j in range(i + 1, n_source):
            st2 = get_bin_spikes(spikes[j], sources.shape[1])
            _, shift = max_xcorr(st1, st2, max_shift=int(max_shift * fsamp))
            tp, _, _ = match_spike_trains(
                st1, st2, shift=shift, tol=tol, fsamp=fsamp
            )
            denom = max(len(spikes[i]), len(spikes[j]))
            match_score = tp / denom if denom > 0 else 0.0
            if match_score >= threshold:
                new_labels[j] = i

    unique_labels = np.unique(new_labels)
    new_sources = np.zeros((len(unique_labels), sources.shape[1]))
    new_spikes: dict[int, np.ndarray] = {i: np.array([]) for i in range(len(unique_labels))}
    new_sil = np.zeros(len(unique_labels))
    new_filters = np.zeros((mu_filters.shape[0], len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = (new_labels == label).astype(int)
        best_idx = int(np.argmax(mask * sil))
        new_sources[i, :] = sources[best_idx, :]
        new_spikes[i] = spikes[best_idx]
        new_sil[i] = sil[best_idx]
        new_filters[:, i] = mu_filters[:, best_idx]

    return new_sources, new_spikes, new_sil, new_filters


def remove_bad_sources(
    sources: np.ndarray,
    spikes: dict[int, np.ndarray],
    sil: np.ndarray,
    mu_filters: np.ndarray,
    threshold: float = 0.9,
    min_num_spikes: int = 10,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Drop sources below a SIL threshold or with too few spikes.

    Args:
        sources: ``(n_mu, n_samples)`` source signals.
        spikes: ``{i: spike_indices}``.
        sil: ``(n_mu,)`` SIL scores.
        mu_filters: ``(n_features, n_mu)`` extraction filters.
        threshold: Minimum SIL to keep a source.
        min_num_spikes: Minimum spike count to keep a source.

    Returns:
        ``(new_sources, new_spikes, new_sil, new_filters)`` with the
        surviving sources, re-indexed starting from 0.
    """
    bad_idx: list[int] = []
    new_spikes: dict[int, np.ndarray] = {}
    new_spike_idx = 0
    for i in range(sources.shape[0]):
        if sil[i] < threshold or len(spikes[i]) < min_num_spikes:
            bad_idx.append(i)
        else:
            new_spikes[new_spike_idx] = spikes[i]
            new_spike_idx += 1

    bad = np.asarray(bad_idx, dtype=int)
    new_sources = np.delete(sources, bad, axis=0)
    new_sil = np.delete(sil, bad, axis=0)
    new_filters = np.delete(mu_filters, bad, axis=1)

    return new_sources, new_spikes, new_sil, new_filters
