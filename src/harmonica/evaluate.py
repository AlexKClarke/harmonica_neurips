"""Spike-train evaluation — matching, scoring, and per-source F1.

Main entry points
-----------------
- :func:`evaluate_decomposition` — multi-source decomposition vs ground-truth
- :func:`shift_aligned_match`  — single pair, returns precision/recall/F1 dict
- :func:`match_spikes`         — low-level pair matcher (seconds)
- :func:`best_time_shift`      — sweep shifts, return best TP count

Units convention
----------------
Spike arrays are sample indices internally. The low-level matching functions
(:func:`match_spikes`, :func:`best_time_shift`) take times in **seconds**.
High-level evaluators (:func:`evaluate_decomposition`, :func:`shift_aligned_match`)
accept sample indices and convert by dividing by ``fsamp``.

Shift alignment corrects for the convolutive delay introduced by Toeplitz
extension — typically 10–50 samples (5–25 ms at 2048 Hz). Always use
:func:`best_time_shift` or :func:`evaluate_decomposition` rather than assuming
a zero shift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import convolve, correlate, correlation_lags, find_peaks
from scipy.stats import kurtosis, skew

# =============================================================================
# Core: spike-train matching
# =============================================================================


def match_spikes(
    s1: np.ndarray,
    s2: np.ndarray,
    shift: float = 0.0,
    tol: float = 0.001,
) -> tuple[int, int, int]:
    """Match two spike trains given a time shift and a matching tolerance.

    Args:
        s1: First neuron's spike times, in seconds.
        s2: Second neuron's spike times, in seconds.
        shift: Delay added to ``s2`` before matching, in seconds.
        tol: Matching window half-width, in seconds. Two spikes match if
            ``|t1 - (t2 + shift)| <= tol``.

    Returns:
        ``(tp, fp, fn)``. ``fp`` counts unmatched ``s1``, ``fn`` counts
        unmatched ``s2``.
    """
    t1 = np.sort(s1)
    t2 = np.sort(s2 + shift)

    matched_1 = np.zeros(len(t1), dtype=bool)
    matched_2 = np.zeros(len(t2), dtype=bool)

    i, j = 0, 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if abs(dt) <= tol:
            matched_1[i] = True
            matched_2[j] = True
            i += 1
            j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1

    tp = int(matched_1.sum())
    fp = int((~matched_1).sum())
    fn = int((~matched_2).sum())
    return tp, fp, fn


def match_spike_trains(
    s1: np.ndarray,
    s2: np.ndarray,
    shift: int = 0,
    tol: float = 0.001,
    fsamp: float = 10000,
) -> tuple[int, int, int]:
    """Match two binary spike trains via convolution with a tolerance kernel.

    Alternative to :func:`match_spikes` that operates on dense binary
    vectors rather than sparse time arrays. Useful when trains are already
    binarised (e.g. for fast bulk comparison).

    Args:
        s1: First binary spike train (0/1), same length as ``s2``.
        s2: Second binary spike train (0/1).
        shift: Integer sample shift applied to ``s2`` via :func:`numpy.roll`.
        tol: Matching window half-width, in seconds.
        fsamp: Sampling frequency, in Hz.

    Returns:
        ``(tp, fp, fn)``.
    """
    s2 = np.roll(s2, shift)

    kernel = np.ones(int(2 * tol * fsamp) + 1)
    masked_s1 = convolve(s1, kernel, mode="same") > 0
    masked_s2 = convolve(s2, kernel, mode="same") > 0

    tp = int(np.logical_and(s1 > 0, masked_s2).sum())
    fp = int(np.logical_and(s1 > 0, ~masked_s2).sum())
    fn = int(np.logical_and(s2 > 0, ~masked_s1).sum())
    return tp, fp, fn


def best_time_shift(
    spikes1: np.ndarray,
    spikes2: np.ndarray,
    tolerance: float = 0.001,
    max_shift: float = 0.01,
    shift_step: float = 0.0005,
) -> tuple[int, int, int, float]:
    """Sweep shifts within ``[-max_shift, +max_shift]`` and return the best.

    The "best" shift is the one that maximises true positives against
    ``spikes1``. Ties keep the first shift tried.

    Args:
        spikes1: First spike train, in seconds.
        spikes2: Second spike train, in seconds.
        tolerance: Matching window half-width, in seconds.
        max_shift: Maximum absolute shift to consider, in seconds.
        shift_step: Step size for the shift sweep, in seconds.

    Returns:
        ``(tp, fp, fn, shift_seconds)``.
    """
    best_tp = 0
    best_shift = 0.0
    best_fp, best_fn = 0, 0

    shifts = np.arange(-max_shift, max_shift + shift_step, shift_step)
    for shift in shifts:
        tp, fp, fn = match_spikes(spikes1, spikes2, shift, tolerance)
        if tp > best_tp:
            best_tp, best_fp, best_fn = tp, fp, fn
            best_shift = float(shift)

    return best_tp, best_fp, best_fn, best_shift


def shift_aligned_match(
    est_sp: np.ndarray,
    gt_sp: np.ndarray,
    fsamp: float,
    tol: float = 0.002,
    max_shift: float = 0.025,
) -> dict[str, Any]:
    """Shift-aligned single-pair matching.

    This is the standard single-source matcher. For multi-source
    evaluation use :func:`evaluate_decomposition`.

    Args:
        est_sp: Estimated spike **sample indices** (not seconds).
        gt_sp:  Ground-truth spike **sample indices** (not seconds).
        fsamp: Sampling frequency, in Hz.
        tol: Matching window half-width, in seconds (default 2 ms).
        max_shift: Maximum absolute shift to consider, in seconds
            (default 25 ms).

    Returns:
        Dict with keys ``tp``, ``fp``, ``fn``, ``precision``, ``recall``,
        ``f1``, ``shift_samples``. The shift convention is
        ``est_aligned = est_sp - shift_samples``.
    """
    est_s = np.asarray(est_sp, dtype=float) / fsamp
    gt_s = np.asarray(gt_sp, dtype=float) / fsamp
    if len(est_s) == 0 or len(gt_s) == 0:
        return {
            "tp": 0, "fp": len(est_s), "fn": len(gt_s),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "shift_samples": 0,
        }
    tp, fp, fn, shift = best_time_shift(
        est_s, gt_s, tolerance=tol, max_shift=max_shift
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "shift_samples": int(round(shift * fsamp)),
    }


# =============================================================================
# Binary spike-train utilities
# =============================================================================


def get_bin_spikes(spike_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """Binary spike train from sample indices.

    Args:
        spike_indices: Integer spike sample indices.
        n_samples: Length of the output vector.

    Returns:
        ``(n_samples,)`` int array with 1 at spike positions.
    """
    spike_train = np.zeros(n_samples, dtype=int)
    spike_train[spike_indices] = 1
    return spike_train


def bin_spikes(
    spike_times: np.ndarray,
    fsamp: float = 10000,
    t_start: float = 0,
    t_end: float = 60,
) -> np.ndarray:
    """Binary spike train from spike times (in seconds).

    Args:
        spike_times: Spike times, in seconds.
        fsamp: Target binning rate, in Hz.
        t_start: Start of the time window, in seconds.
        t_end: End of the time window, in seconds.

    Returns:
        Binary spike train covering ``[t_start, t_end]``.
    """
    step_size = 1 / fsamp
    n_samples = int(np.ceil((t_end - t_start) / step_size)) + 1
    spike_train = np.zeros(n_samples, dtype=int)
    spike_indices = np.round((spike_times - t_start) / step_size).astype(int)
    spike_indices = spike_indices[
        (spike_indices >= 0) & (spike_indices < n_samples)
    ]
    spike_train[spike_indices] = 1
    return spike_train


def max_xcorr(
    sig1: np.ndarray,
    sig2: np.ndarray,
    max_shift: int = 1000,
) -> tuple[float, int]:
    """Peak of the cross-correlation between two signals.

    Args:
        sig1: Reference signal.
        sig2: Comparison signal.
        max_shift: Maximum absolute lag to consider, in samples.

    Returns:
        ``(peak_corr, best_shift_samples)``.
    """
    corr = correlate(sig1, sig2, mode="full")
    lags = correlation_lags(len(sig1), len(sig2), mode="full")
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr_win = corr[mask]
    lags_win = lags[mask]
    best_idx = int(np.argmax(corr_win))
    return float(corr_win[best_idx]), int(lags_win[best_idx])


def label_sources(
    df: pd.DataFrame,
    fsamp: float = 10000,
    t_start: float = 0,
    t_end: float = 60,
    threshold: float = 0.3,
    max_shift: float = 0.1,
    tol: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster source ids by mutual spike-train matching score.

    Builds a full pairwise match-score matrix between all sources in ``df``
    and relabels duplicates so that each cluster shares the label of its
    lowest-indexed member.

    Args:
        df: Long-format table with columns ``source_id`` and ``spike_time``
            (spike times in seconds).
        fsamp: Binning rate used internally, in Hz.
        t_start: Start of the time window, in seconds.
        t_end: End of the time window, in seconds.
        threshold: Minimum pairwise match score to merge two sources.
        max_shift: Maximum absolute xcorr lag, in seconds.
        tol: Matching window half-width, in seconds.

    Returns:
        ``(labels, match_matrix)``. ``labels`` is a 1D array indexed by the
        sorted unique ``source_id`` list; ``match_matrix`` is the full
        symmetric pairwise match score matrix.
    """
    units = sorted(df["source_id"].unique())
    n_source = len(units)
    labels = np.arange(n_source)
    match_matrix = np.identity(n_source)

    for i in range(n_source):
        spikes_1 = df[df["source_id"] == units[i]]["spike_time"].values
        st1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)

        for j in range(i + 1, n_source):
            spikes_2 = df[df["source_id"] == units[j]]["spike_time"].values
            st2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _, shift = max_xcorr(st1, st2, max_shift=int(max_shift * fsamp))
            tp, _, _ = match_spikes(
                spikes_1, spikes_2, shift=shift / fsamp, tol=tol
            )
            denom = max(len(spikes_1), len(spikes_2))
            match_score = tp / denom if denom > 0 else 0.0

            match_matrix[i, j] = match_score
            match_matrix[j, i] = match_score

            if match_score >= threshold:
                labels[j] = i

    return labels, match_matrix


# =============================================================================
# Quality metrics
# =============================================================================


def signal_based_quality_metrics(
    source: np.ndarray,
    spikes: np.ndarray,
    fsamp: float,
    min_peak_dist: float = 0.01,
    match_dist: float = 0.001,
) -> dict[str, float]:
    """Batch of signal-based quality metrics for a source / spike pair.

    Returns a dict with ``n_spikes``, ``skew_val``, ``kurt_val``,
    ``peak_height``, ``cov_peak``, ``sil``, ``sep_prctile90``, ``sep_std``,
    ``pnr``, ``z_score_height``.

    Args:
        source: The source signal, ``(T,)``.
        spikes: Spike sample indices into ``source``.
        fsamp: Sampling frequency, in Hz.
        min_peak_dist: Minimum peak separation for background peak detection,
            in seconds.
        match_dist: Exclusion window half-width around each spike when
            selecting background peaks, in seconds.
    """
    qm: dict[str, float] = {}

    qm["n_spikes"] = len(spikes)
    qm["skew_val"] = float(skew(source))
    qm["kurt_val"] = float(kurtosis(source))
    qm["peak_height"] = float(np.mean(source[spikes]))
    qm["cov_peak"] = float(np.std(source[spikes]) / np.mean(source[spikes]))

    sil, background_peaks = pseudo_sil_score(
        source, spikes, fsamp,
        min_peak_dist=min_peak_dist, match_dist=match_dist,
    )
    qm["sil"] = sil

    pred_amps = source[spikes]
    back_amps = source[background_peaks]

    qm["sep_prctile90"] = float(
        (np.percentile(pred_amps, 10) - np.percentile(back_amps, 90))
        / np.mean(pred_amps)
    )
    qm["sep_std"] = float(
        (np.mean(pred_amps) - np.mean(back_amps)) / np.std(pred_amps)
    )

    pnr, noise_indices = calc_pnr(source, spikes)
    qm["pnr"] = pnr
    qm["z_score_height"] = float(
        np.mean(source[spikes]) / np.std(source[noise_indices])
    )

    return qm


def pseudo_sil_score(
    source: np.ndarray,
    spikes: np.ndarray,
    fsamp: float,
    min_peak_dist: float = 0.01,
    match_dist: float = 0.001,
) -> tuple[float, np.ndarray]:
    """Silhouette-like score between spike amplitudes and background peaks.

    The background set is the set of peaks of ``source * |source|`` that fall
    outside a ``±match_dist`` window around each predicted spike.

    Args:
        source: Source signal, ``(T,)``.
        spikes: Predicted spike sample indices.
        fsamp: Sampling frequency, in Hz.
        min_peak_dist: Minimum distance for peak detection, in seconds.
        match_dist: Exclusion window half-width around predicted spikes,
            in seconds.

    Returns:
        ``(sil, background_spikes)``. ``sil ∈ [0, 1]``-ish,
        ``background_spikes`` is an array of sample indices.
    """
    spikes = np.asarray(spikes, dtype=int)
    match_window = int(round(fsamp * match_dist))
    min_dist = int(round(fsamp * min_peak_dist))

    sig = source * np.abs(source)
    detected_peaks, _ = find_peaks(sig, distance=min_dist)

    mask = np.ones(len(detected_peaks), dtype=bool)
    for spike in spikes:
        mask &= np.abs(detected_peaks - spike) > match_window
    background_spikes = detected_peaks[mask]

    if len(spikes) < 2 or len(background_spikes) == 0:
        # Return a tuple here too: callers tuple-unpack the result
        # (e.g. the call in signal_based_quality_metrics).
        return 0.0, background_spikes

    pred_amps = sig[spikes].reshape(-1, 1)
    back_amps = sig[background_spikes].reshape(-1, 1)

    centroids = [np.mean(pred_amps), np.mean(back_amps)]
    within = np.sum((pred_amps - centroids[0]) ** 2) if len(pred_amps) > 1 else 0.0
    between = np.sum((pred_amps - centroids[1]) ** 2)

    denom = max(between, within)
    sil = float((between - within) / denom) if denom > 0 else 0.0
    return sil, background_spikes


def calc_pnr(
    source: np.ndarray,
    spikes_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Pulse-to-noise ratio (dB) of the spike cluster vs the noise cluster.

    Args:
        source: Source signal, ``(T,)``.
        spikes_idx: Spike sample indices.

    Returns:
        ``(pnr_db, noise_indices)``.
    """
    signal_length = len(source)
    sig = source * np.abs(source)

    expanded = set()
    for idx in spikes_idx:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < signal_length:
                expanded.add(neighbor)
    expanded_indices = np.array(sorted(expanded))
    noise_indices = np.setdiff1d(np.arange(signal_length), expanded_indices)

    peak_cluster = sig[expanded_indices] ** 2
    noise_cluster = sig[noise_indices] ** 2
    pnr = 10 * np.log10(np.mean(peak_cluster) / np.mean(noise_cluster))
    return float(pnr), noise_indices


def get_basic_spike_statistics(
    spike_times: np.ndarray,
    min_num_spikes: int = 10,
) -> tuple[float, float]:
    """Mean firing rate (Hz) and ISI coefficient of variation.

    Rejects ISIs above ``10 * std(isi)`` to ignore quiescent periods.

    Args:
        spike_times: Spike times, in seconds.
        min_num_spikes: Minimum spike count required to compute stats.

    Returns:
        ``(cov, mean_fr)``. Returns ``(inf, 0)`` if there are too few spikes.
    """
    cov = np.inf
    mean_fr = 0.0

    if len(spike_times) > min_num_spikes:
        isi = np.diff(spike_times)
        std_isi = np.std(isi)
        isi = isi[isi <= 10 * std_isi]
        if len(isi) > 1:
            cov = float(np.std(isi) / np.mean(isi))
            mean_fr = float(1 / np.mean(isi))
    return cov, mean_fr


# =============================================================================
# Main decomposition evaluators
# =============================================================================


def evaluate_decomposition(
    est_spikes: dict[Any, np.ndarray],
    gt_spikes: dict | list,
    fsamp: float,
    tol: float = 0.002,
    max_shift: float = 0.025,
) -> dict[str, Any]:
    """Multi-source decomposition evaluation against ground truth.

    Matches each estimated source to its best-F1 GT source using
    shift-aligned matching, then aggregates.

    Args:
        est_spikes: ``{source_id: spike_indices}``. Values are sample indices.
        gt_spikes: Either a dict ``{gt_id: spike_indices}`` or a list of
            spike index arrays.
        fsamp: Sampling frequency, in Hz.
        tol: Matching window half-width, in seconds.
        max_shift: Maximum absolute alignment shift, in seconds.

    Returns:
        Dict with:
            - ``per_source``: list of ``{tp, fp, fn, precision, recall, f1,
              est_id, gt_id}`` for each estimated source.
            - ``n_gt_found``: count of GT sources with best-match F1 > 0.5.
            - ``n_gt_total``: total GT source count.
            - ``mean_f1``: mean F1 across estimated sources with F1 > 0.
            - ``gt_avg_f1``: mean best-F1 across **all** GT sources (including
              unfound ones, contributing 0).
    """
    if isinstance(gt_spikes, dict):
        gt_list = [np.asarray(gt_spikes[k]) for k in sorted(gt_spikes.keys())]
    else:
        gt_list = [np.asarray(s) for s in gt_spikes]

    est_ids = sorted(est_spikes.keys())
    n_gt = len(gt_list)

    per_source: list[dict[str, Any]] = []
    gt_best_f1 = np.zeros(n_gt)

    for e_id in est_ids:
        e_sp = np.asarray(est_spikes[e_id]) / fsamp

        best = {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 0, "recall": 0, "f1": 0,
            "est_id": e_id, "gt_id": -1,
        }

        for g_idx, g_sp_raw in enumerate(gt_list):
            g_sp = np.asarray(g_sp_raw) / fsamp
            tp, fp, fn, _ = best_time_shift(
                e_sp, g_sp, tolerance=tol, max_shift=max_shift
            )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            if f1 > best["f1"]:
                best = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": precision, "recall": recall, "f1": f1,
                    "est_id": e_id, "gt_id": g_idx,
                }

        per_source.append(best)
        if best["gt_id"] >= 0:
            gt_best_f1[best["gt_id"]] = max(
                gt_best_f1[best["gt_id"]], best["f1"]
            )

    n_gt_found = int(np.sum(gt_best_f1 > 0.5))
    matched = [r["f1"] for r in per_source if r["f1"] > 0]
    mean_f1 = float(np.mean(matched)) if matched else 0.0
    gt_avg_f1 = float(np.mean(gt_best_f1)) if n_gt > 0 else 0.0

    return {
        "per_source": per_source,
        "n_gt_found": n_gt_found,
        "n_gt_total": n_gt,
        "mean_f1": mean_f1,
        "gt_avg_f1": gt_avg_f1,
    }


def evaluate_windowed(
    est_spikes: dict[Any, np.ndarray],
    gt_spikes: dict | list,
    fsamp: float,
    windows: list[tuple[int, int]],
    tol: float = 0.002,
    max_shift: float = 0.050,
) -> dict[str, Any]:
    """Decomposition evaluation with per-window shift alignment.

    Useful when the convolutive delay varies across windows (e.g. different
    poses cause different ICA filter delays). Finds the best shift per
    window per MU, then aggregates TP/FP/FN across windows.

    Args:
        est_spikes: ``{source_id: spike_indices}`` (sample indices).
        gt_spikes: dict ``{gt_id: spike_indices}`` or list of arrays.
        fsamp: Sampling frequency, in Hz.
        windows: List of ``(start_sample, end_sample)`` tuples.
        tol: Matching window half-width, in seconds.
        max_shift: Maximum absolute shift per window, in seconds.

    Returns:
        Same shape as :func:`evaluate_decomposition` (minus ``gt_avg_f1``).
    """
    if isinstance(gt_spikes, dict):
        gt_list = [np.asarray(gt_spikes[k]) for k in sorted(gt_spikes.keys())]
    else:
        gt_list = [np.asarray(s) for s in gt_spikes]

    est_ids = sorted(est_spikes.keys())
    n_gt = len(gt_list)

    per_source: list[dict[str, Any]] = []
    gt_best_f1 = np.zeros(n_gt)

    for e_id in est_ids:
        e_sp = np.asarray(est_spikes[e_id])

        best = {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 0, "recall": 0, "f1": 0,
            "est_id": e_id, "gt_id": -1,
        }

        for g_idx, g_sp_raw in enumerate(gt_list):
            g_sp = np.asarray(g_sp_raw)

            total_tp, total_fp, total_fn = 0, 0, 0
            for w_start, w_end in windows:
                e_mask = (e_sp >= w_start) & (e_sp < w_end)
                g_mask = (g_sp >= w_start) & (g_sp < w_end)
                e_w = e_sp[e_mask] / fsamp
                g_w = g_sp[g_mask] / fsamp

                if len(e_w) == 0 and len(g_w) == 0:
                    continue
                if len(e_w) == 0:
                    total_fn += len(g_w)
                    continue
                if len(g_w) == 0:
                    total_fp += len(e_w)
                    continue

                tp, fp, fn, _ = best_time_shift(
                    e_w, g_w, tolerance=tol, max_shift=max_shift
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            if f1 > best["f1"]:
                best = {
                    "tp": total_tp, "fp": total_fp, "fn": total_fn,
                    "precision": precision, "recall": recall, "f1": f1,
                    "est_id": e_id, "gt_id": g_idx,
                }

        per_source.append(best)
        if best["gt_id"] >= 0:
            gt_best_f1[best["gt_id"]] = max(
                gt_best_f1[best["gt_id"]], best["f1"]
            )

    n_gt_found = int(np.sum(gt_best_f1 > 0.5))
    matched = [r["f1"] for r in per_source if r["f1"] > 0]
    mean_f1 = float(np.mean(matched)) if matched else 0.0

    return {
        "per_source": per_source,
        "n_gt_found": n_gt_found,
        "n_gt_total": n_gt,
        "mean_f1": mean_f1,
    }


def evaluate_quasi_static_by_segment(
    est_spikes: dict[Any, np.ndarray],
    gt_spikes: list,
    segment_boundaries: list[tuple[int, int]],
    fsamp: float,
    tol: float = 0.002,
) -> tuple[list[dict], dict]:
    """Per-segment decomposition evaluation.

    Slices both estimated and ground-truth spikes into each segment
    (indices are rebased to start at 0 within the segment), runs
    :func:`evaluate_decomposition` inside each, and also returns the
    whole-recording aggregate.

    Args:
        est_spikes: ``{source_id: spike_indices}``.
        gt_spikes: List of spike index arrays, one per GT MU.
        segment_boundaries: List of ``(start_sample, end_sample)`` tuples.
        fsamp: Sampling frequency, in Hz.
        tol: Matching window half-width, in seconds.

    Returns:
        ``(per_segment, overall)`` where ``per_segment`` is a list of
        evaluation dicts (each with an added ``segment`` key) and
        ``overall`` is the full-recording evaluation.
    """
    per_segment: list[dict] = []

    for seg_start, seg_end in segment_boundaries:
        seg_est: dict[Any, np.ndarray] = {}
        for sid, sp in est_spikes.items():
            sp = np.asarray(sp)
            mask = (sp >= seg_start) & (sp < seg_end)
            local_sp = sp[mask] - seg_start
            if len(local_sp) > 0:
                seg_est[sid] = local_sp

        seg_gt = []
        for g_sp in gt_spikes:
            g_sp = np.asarray(g_sp)
            mask = (g_sp >= seg_start) & (g_sp < seg_end)
            seg_gt.append(g_sp[mask] - seg_start)

        result = evaluate_decomposition(seg_est, seg_gt, fsamp, tol=tol)
        result["segment"] = (seg_start, seg_end)
        per_segment.append(result)

    overall = evaluate_decomposition(est_spikes, gt_spikes, fsamp, tol=tol)
    return per_segment, overall
