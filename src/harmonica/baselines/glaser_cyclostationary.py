"""Glaser & Holobar 2018/2019 — paper-faithful Cyclostationary CKC.

Paper: *Motor Unit Identification from High-Density Surface Electromyograms
in Repeated Dynamic Muscle Contractions*, IEEE TNSRE 27(1):66-75, 2019,
DOI ``10.1109/TNSRE.2018.2885283``.

This module implements the full **Pseudocode 1** algorithm from §II-C — the
PNR-gated reference-anchored walk that the simplified
:mod:`harmonica.baselines.glaser` skipped.

Algorithm (per MU seed):

    1. On a short reference window ``[a1, a2]`` (typ. 2 s), run classical CKC
       (we use ``run_ica_loop`` with peel-off as the seeder) and pick a
       seed separation vector that yields a clean spike-train estimate.
    2. Compute initial ``c_tjy_ref = (1/|a|) Σ t̂j(n) y(n)`` over ``[a1, a2]``
       (paper Eq. 8).
    3. Forward-walk in steps of ``a`` ≈ ramp/5:
         - On each candidate window ``[b1, b1+a]`` form the merged interval
           ``Υ = [a1, a2] ∪ [b1, b1+a]``;
         - evaluate ``t̂j(n) = c_tjy_ref^T Cy_Υ^{-1} y(n)`` (Eq. 9);
         - compute local PNR on the candidate portion (Eq. 7);
         - if ``PNR >= 28 dB``, accept the window: accumulate spike-weighted
           ``c_tjy_acc += Σ_b t̂j(n) y(n)`` and add the window to Υ.
    4. Normalise ``c_tjy = (c_tjy_ref + c_tjy_acc) / ‖·‖`` and apply Eq. 5
       to estimate the spike train on the entire signal.
    5. Right-shift and left-shift refinements (paper steps 9, 10) repeated
       up to ``max_shift_iters`` times each.

This implementation skips a few minor knobs (``f`` ramp shape, optimal step
size per MU) but matches the algorithmic core that distinguishes the paper
from the simplified segmented variant.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening


# =============================================================================
# Helpers
# =============================================================================


def _local_pnr(t_hat: np.ndarray) -> float:
    """Eq. 7: 10·log10(E[t̂² | spike] / E[t̂² | null]).

    Two-cluster k-means on |t̂| splits samples into spike/null; PNR is the
    log-ratio of mean squared values in each cluster. Returns 0.0 if either
    cluster is empty.
    """
    if len(t_hat) < 20 or t_hat.std() < 1e-9:
        return 0.0
    t_abs = np.abs(t_hat).reshape(-1, 1)
    try:
        km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(t_abs)
    except Exception:
        return 0.0
    centers = km.cluster_centers_.flatten()
    spike_label = int(np.argmax(centers))
    spike_mask = km.labels_ == spike_label
    null_mask = ~spike_mask
    if spike_mask.sum() < 3 or null_mask.sum() < 3:
        return 0.0
    e_spike = float(np.mean(t_hat[spike_mask] ** 2))
    e_null = float(np.mean(t_hat[null_mask] ** 2))
    if e_null <= 1e-12:
        return 0.0
    return 10.0 * np.log10(e_spike / e_null)


def _t_hat_eq9(c_tjy: np.ndarray, y: np.ndarray, union_idx: np.ndarray,
               eval_idx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Eq. 9: ``t̂j(n) = c_tjy^T Cy_Υ^{-1} y(n)`` evaluated on ``eval_idx``.

    ``Cy_Υ`` is the empirical correlation over ``y[:, union_idx]``.
    """
    Y_u = y[:, union_idx]
    Cy = (Y_u @ Y_u.T) / max(Y_u.shape[1], 1)
    Cy_reg = Cy + eps * np.eye(Cy.shape[0]) * np.trace(Cy) / max(Cy.shape[0], 1)
    try:
        Cy_inv = np.linalg.pinv(Cy_reg)
    except np.linalg.LinAlgError:
        return np.zeros(len(eval_idx))
    w_eff = c_tjy @ Cy_inv                      # (D,)
    Y_eval = y[:, eval_idx]                       # (D, |eval|)
    return w_eff @ Y_eval                         # (|eval|,)


def _spike_weight(t_hat: np.ndarray) -> np.ndarray:
    """Threshold-then-square version of t̂ used for c_tjy accumulation.

    Paper Eq. 8 sums ``t_j(n) y(n)`` where t_j is binary (1 at spikes, 0
    otherwise). We approximate by the soft thresholded square — squaring
    suppresses noise samples and emphasises the high-amplitude (spike-like)
    samples, matching the spirit of Eq. 10 with the simple identity window.
    """
    return t_hat ** 2


def _peak_threshold(t_hat: np.ndarray, min_separation_ratio: float = 2.0) -> float:
    """K-means split on |t̂| with separation check.

    If the spike cluster's centroid is < ``min_separation_ratio`` × the null
    centroid (i.e., the clusters aren't well separated → noisy / poorly-
    tracked source), fall back to the 99th percentile so we don't dump
    half the samples into "spike". This gates the 700-spike-per-MU
    over-detection seen on low-PNR walks.
    """
    if len(t_hat) < 20:
        return float("inf")
    t_abs = np.abs(t_hat).reshape(-1, 1)
    try:
        km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(t_abs)
        centers = sorted(km.cluster_centers_.flatten())
        if centers[0] <= 1e-9 or centers[1] / centers[0] < min_separation_ratio:
            return float(np.percentile(np.abs(t_hat), 99.0))
        return float(0.5 * (centers[0] + centers[1]))
    except Exception:
        return float(np.percentile(np.abs(t_hat), 99.0))


def _detect_spikes(t_hat: np.ndarray, fs: float,
                   refractory_ms: float = 5.0,
                   threshold: float | None = None) -> np.ndarray:
    """Peak-detect spikes from a continuous spike-train estimate."""
    from scipy.signal import find_peaks
    if threshold is None:
        threshold = _peak_threshold(t_hat)
    if not np.isfinite(threshold) or threshold <= 0:
        return np.array([], dtype=int)
    min_dist = max(1, int(refractory_ms * 1e-3 * fs))
    peaks, _ = find_peaks(t_hat, height=threshold, distance=min_dist)
    return peaks.astype(int)


def _preprocess(emg: np.ndarray, R: int, d_eff: int):
    """Toeplitz-extend → PCA → ZCA whiten → also keep extended (un-whitened) view."""
    z_ext = extension(emg, R)                        # (M*R, T)
    d = min(d_eff, z_ext.shape[0] - 1)
    pca = PCA(n_components=d).fit(z_ext.T)
    z_pca = pca.transform(z_ext.T).T
    z_w, _ = whitening(z_pca, method="ZCA")
    # We work in the whitened space throughout — Cy_Υ-inv simplifies because
    # the global Cz_w is approximately I, but per-window correlation is not I
    # so we still recompute it. Use z_w as ``y`` in the paper notation.
    return z_w


# =============================================================================
# One-MU walk
# =============================================================================


def _walk_one_mu(
    z_w: np.ndarray,
    w_seed: np.ndarray,
    fs: float,
    a1: int, a2: int,
    step: int,
    pnr_threshold: float,
    refractory_ms: float,
    verbose: bool = False,
) -> tuple[np.ndarray, set[int]]:
    """Run the forward PNR-gated walk for a single MU seed.

    Returns the final separation filter (after Eq. 5 normalisation) applied to
    the whole signal, plus the set of accepted sample indices Υ.
    """
    D, T = z_w.shape
    a_idx = np.arange(a1, a2)

    # Step 2-3: classical CKC seed → t̂j on reference, then c_tjy_ref via Eq. 8
    t_ref = w_seed @ z_w[:, a1:a2]
    weights = _spike_weight(t_ref)
    if weights.sum() < 1e-9:
        return np.zeros(T), set()
    c_tjy_ref = (weights[None, :] * z_w[:, a1:a2]).sum(axis=1) / weights.sum()

    # Forward walk
    accepted = set(int(i) for i in a_idx)
    c_tjy_acc = np.zeros(D)
    n_acc_windows = 0
    b1 = 0
    while b1 + step <= T:
        # Skip any window overlapping the reference
        if not (b1 + step <= a1 or b1 >= a2):
            b1 += step; continue

        b_idx = np.arange(b1, b1 + step)
        union_idx = np.r_[a_idx, b_idx]

        t_hat_union = _t_hat_eq9(c_tjy_ref, z_w, union_idx, union_idx)
        t_hat_b = t_hat_union[len(a_idx):]
        pnr = _local_pnr(t_hat_b)

        if pnr >= pnr_threshold:
            w_b = _spike_weight(t_hat_b)
            if w_b.sum() > 1e-9:
                c_tjy_acc += (w_b[None, :] * z_w[:, b1:b1+step]).sum(axis=1) / w_b.sum()
                n_acc_windows += 1
            accepted.update(int(i) for i in b_idx)
        b1 += step

    # Combine reference + accumulated; normalise and apply globally (Eq. 5)
    if n_acc_windows == 0:
        c_tjy_total = c_tjy_ref
    else:
        c_tjy_total = (c_tjy_ref + c_tjy_acc) / (1 + n_acc_windows)
    norm = np.linalg.norm(c_tjy_total) + 1e-12
    c_tjy_total = c_tjy_total / norm

    # Final estimate over the full signal — clamp accepted to valid range
    T_total = z_w.shape[1]
    union_idx = np.array(sorted(i for i in accepted if 0 <= i < T_total))
    if len(union_idx) < 100:
        # Too few accepted samples — fall back to seed filter
        t_full = w_seed @ z_w
    else:
        t_full = _t_hat_eq9(c_tjy_total, z_w, union_idx,
                             np.arange(T))

    if verbose:
        print(f"    walk: ref [{a1}:{a2}] +{n_acc_windows} accepted windows "
              f"({len(accepted)/T*100:.1f}% coverage)")
    return t_full, accepted


def _shift_step(
    z_w: np.ndarray,
    t_full_prev: np.ndarray,
    accepted: set[int],
    fs: float,
    step: int,
    pnr_threshold: float,
    direction: str,           # 'right' or 'left'
    verbose: bool = False,
) -> tuple[np.ndarray | None, set[int] | None]:
    """Right- or left-shift refinement (paper steps 9, 10).

    Find the latest (right) or earliest (left) accepted edge ``c`` and
    re-anchor the reference around it via a linear ramp window.
    """
    if not accepted:
        return None, None
    accepted_sorted = sorted(accepted)
    if direction == "right":
        # Find sample c such that c+1 is not accepted
        c = None
        for s in reversed(accepted_sorted):
            if (s + 1) not in accepted:
                c = s; break
        if c is None or c < step:
            return None, None
        nc = np.arange(c - step, c)
        ramp = np.clip(0.5 - (c - step - nc) / step, 0.0, 1.0)
    else:
        c = None
        for s in accepted_sorted:
            if (s - 1) not in accepted:
                c = s; break
        if c is None or c + step >= z_w.shape[1]:
            return None, None
        nc = np.arange(c, c + step)
        ramp = np.clip(0.5 + (c + step - nc) / step, 0.0, 1.0)
    t_nc = t_full_prev[nc]
    weighted = ramp * t_nc
    weights2 = weighted ** 2
    if weights2.sum() < 1e-9:
        return None, None
    c_tjy_shift = (weights2[None, :] * z_w[:, nc]).sum(axis=1) / weights2.sum()

    a1_new, a2_new = int(c - step // 2), int(c + step // 2)
    a1_new = max(0, a1_new); a2_new = min(z_w.shape[1], a2_new)

    # Normalise and apply Eq. 5
    norm = np.linalg.norm(c_tjy_shift) + 1e-12
    c_tjy_shift = c_tjy_shift / norm
    union_idx = np.array(sorted(
        i for i in (accepted | set(range(a1_new, a2_new))) if 0 <= i < z_w.shape[1]
    ))
    t_full = _t_hat_eq9(c_tjy_shift, z_w, union_idx, np.arange(z_w.shape[1]))

    new_accepted = set(accepted)
    T_total = z_w.shape[1]
    if direction == "right":
        new_accepted.update(i for i in range(c + 1, c + step // 2 + 1) if 0 <= i < T_total)
    else:
        new_accepted.update(i for i in range(c - step // 2, c) if 0 <= i < T_total)
    if verbose:
        print(f"    shift {direction}: new ref [{a1_new}:{a2_new}], "
              f"coverage {len(new_accepted)/z_w.shape[1]*100:.1f}%")
    return t_full, new_accepted


# =============================================================================
# Public entry point
# =============================================================================


def run_glaser_cyclostationary(
    emg: np.ndarray,
    fs: float,
    ref_window_s: float = 2.0,
    step_s: float = 0.33,
    pnr_threshold: float = 28.0,
    R: int = 4,
    d_eff: int = 128,
    sil_th: float = 0.7,
    n_seed_iters: int = 60,
    refractory_ms: float = 5.0,
    seed: int = 1909,
    max_shift_iters: int = 2,
    min_pnr_keep: float = 10.0,
    min_coverage: float = 0.10,
    max_firing_rate_hz: float = 35.0,  # reject MUs with non-physiological rates
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Run the cyclostationary CKC method on a dynamic hdEMG recording.

    Args:
        emg: ``(M, T)`` raw EMG.
        fs: Sampling frequency (Hz).
        ref_window_s: Length of the initial reference interval ``[a1, a2]``
            (paper §II-B uses 1-2 s).
        step_s: Step size ``a`` for the forward walk (paper: ramp/5).
            For our 0.3 Hz recordings (cycle 3.33 s, half-cycle ramp
            1.67 s) this defaults to 0.33 s.
        pnr_threshold: PNR gate in dB (paper: 28 dB).
        R, d_eff: Toeplitz-extension factor and PCA target dim (cBSS knobs).
        sil_th: Initial CKC silhouette threshold for picking seed filters
            on the reference window.
        n_seed_iters: ``run_ica_loop`` iteration cap.
        refractory_ms: Spike-detection refractory floor.
        seed: RNG seed.
        max_shift_iters: Max right-shift + left-shift refinement passes.
        min_pnr_keep: After all walks, drop final spike trains whose global
            PNR (computed over Υ) is below this threshold.
        verbose: Per-MU progress logging.

    Returns:
        ``{mu_id: spike_indices}`` directly consumable by
        :func:`harmonica.evaluate.evaluate_decomposition`.
    """
    M, T = emg.shape
    a1 = 0
    a2 = min(int(ref_window_s * fs), T)
    step = max(int(step_s * fs), 32)

    # Preprocess full signal
    z_w = _preprocess(emg, R, d_eff)
    D = z_w.shape[0]

    # Seed: classical CKC on the reference window → seed filters
    z_w_ref = z_w[:, a1:a2]
    try:
        _src, _spk, sils, filters = run_ica_loop(
            z_w_ref, fs,
            sil_th=sil_th, n_iter=n_seed_iters,
            seed=seed, refinement_loop=True, peel_off=True,
        )
    except Exception as exc:
        if verbose:
            print(f"  [glaser-cyclo] seed CKC failed: {exc}")
        return {}

    n_seed_mus = filters.shape[1]
    if verbose:
        print(f"[glaser-cyclo] {n_seed_mus} seed MUs from CKC on [{a1}:{a2}]")

    result: dict[int, np.ndarray] = {}
    out_id = 0
    for j in range(n_seed_mus):
        w_seed = filters[:, j] / (np.linalg.norm(filters[:, j]) + 1e-12)
        # Forward walk
        t_full, accepted = _walk_one_mu(
            z_w, w_seed, fs, a1, a2, step, pnr_threshold, refractory_ms,
            verbose=verbose,
        )
        if len(accepted) == 0:
            continue
        # Shifts
        for it in range(max_shift_iters):
            ret_r = _shift_step(z_w, t_full, accepted, fs, step, pnr_threshold, "right", verbose)
            if ret_r[0] is not None:
                t_full, accepted = ret_r
            ret_l = _shift_step(z_w, t_full, accepted, fs, step, pnr_threshold, "left", verbose)
            if ret_l[0] is not None:
                t_full, accepted = ret_l

        # Final filter quality gate on the full signal
        coverage = len(accepted) / T
        # Sign correction: peaks should be positive. Decide by looking at top quantile.
        if np.percentile(t_full, 99.5) < -np.percentile(t_full, 0.5):
            t_full = -t_full
        global_pnr = _local_pnr(t_full)
        if global_pnr < min_pnr_keep or coverage < min_coverage:
            if verbose:
                print(f"  MU{j}: dropped — coverage={coverage:.1%}, PNR={global_pnr:.1f}")
            continue

        spikes = _detect_spikes(t_full, fs, refractory_ms=refractory_ms)
        if len(spikes) < 5:
            continue
        rate = len(spikes) / (T / fs)
        if rate > max_firing_rate_hz:
            if verbose:
                print(f"  MU{j}: dropped — non-physiological rate {rate:.0f} Hz")
            continue
        result[out_id] = spikes
        if verbose:
            print(f"  MU{j} → out{out_id}: coverage={coverage:.1%}, PNR={global_pnr:.1f} dB, n_spikes={len(spikes)}")
        out_id += 1

    return result
