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
from scipy.signal import convolve, correlate, correlation_lags

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
            - ``gt_best_f1``: ``(n_gt,)`` array, best F1 per GT source in
              sorted-key order (0.0 for GT sources no estimate matched).
            - ``gt_best_recall``: ``(n_gt,)`` array, the recall of the same
              argmax-F1 match (not an independently maximised recall).
    """
    if isinstance(gt_spikes, dict):
        gt_list = [np.asarray(gt_spikes[k]) for k in sorted(gt_spikes.keys())]
    else:
        gt_list = [np.asarray(s) for s in gt_spikes]

    est_ids = sorted(est_spikes.keys())
    n_gt = len(gt_list)

    per_source: list[dict[str, Any]] = []
    gt_best_f1 = np.zeros(n_gt)
    gt_best_recall = np.zeros(n_gt)

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
        if best["gt_id"] >= 0 and best["f1"] > gt_best_f1[best["gt_id"]]:
            gt_best_f1[best["gt_id"]] = best["f1"]
            gt_best_recall[best["gt_id"]] = best["recall"]

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
        "gt_best_f1": gt_best_f1,
        "gt_best_recall": gt_best_recall,
    }
