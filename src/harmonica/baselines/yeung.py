"""Yeung et al. 2024 — Adaptive HD-sEMG decoding baseline.

Paper: *Adaptive HD-sEMG decomposition: towards robust real-time
decoding of neural drive*, J Neural Eng 21(2):026012, 2024.
DOI `10.1088/1741-2552/ad33b0`.

Port of Algorithm 1 (§2.3). Core ideas:

- Maintain ``μ̃z`` (mean), ``Σ̃z z̃`` (covariance) of extended obs. with
  EMA forgetting factor ``λ``.
- Source: ``ŝ(k) = v^T Σ_{z̃z̃}^{-1} (z̃(k) − μ̃z)`` — no explicit ZCA.
- ``Ψ_mem``: FIFO buffer of cardinality ``N`` of extended observations
  at recent spike times. Filter updated via
  ``v = mean(z̃_Ψ − μ̃z*)`` over the buffer (Eq. 9).
- Spike detection: z-score threshold ``τ_z`` on the rectified peak
  amplitudes of each window.
- Spike centroid ``c_hi = (v^T Σ^{*-1} v)²`` (Eq. 10); noise centroid
  updated with the forgetting factor (Eq. 11).

The peaks above ``τ_z`` serve twice: they are appended to ``Ψ_mem`` (and so
update the filter) and they are the reported spikes. There is no separate
output decision, e.g. against the centroids ``c_hi`` / ``c_lo``, which are
tracked but not used for detection.

Paper parameter values: ``λ = 0.985``, ``|Ψ_mem| = 110``, ``τ_z = 3.3``.
"""
from __future__ import annotations

import collections

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten


# =============================================================================
# Public entry point
# =============================================================================


def run_yeung(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 1.5,
    batch_ms: float = 200.0,
    ext_fact: int = 10,
    d_eff: int = 128,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    forgetting: float = 0.985,
    psi_size: int = 110,
    z_threshold: float = 2.5,
    min_dist_ms: float = 10.0,
    cov_reg: float = 1e-3,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Yeung 2024 adaptive decoding — FastICA calibration → online update.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window, in seconds.
        batch_ms: Online window, in ms.
        ext_fact: Toeplitz extension factor.
        d_eff: PCA dim cap after extension.
        sil_th: Calibration SIL threshold.
        ica_seed: Calibration RNG seed.
        forgetting: EMA forgetting factor ``λ`` (paper default 0.985).
        psi_size: Capacity of the observation buffer ``|Ψ_mem|``
            (paper default 110).
        z_threshold: z-score cutoff on the rectified peak amplitudes of
            each window. The peaks above it are both added to ``Ψ_mem``
            and returned as spikes. The paper uses 3.3; the default here is
            2.5, and ``experiments/table1_baselines.sh`` runs 2.0.
        min_dist_ms: Min inter-peak spacing, in ms.
        cov_reg: Ridge regulariser on ``Σ*`` before inversion.
        verbose: Per-stage progress.

    Returns:
        ``{mu_id: sorted global spike sample indices}``: the calibration
        spikes followed by the z-gated peaks of every online window,
        deduplicated at 10 ms; units with fewer than 5 spikes are dropped.
    """
    C, T = emg.shape
    emg = np.asarray(emg, dtype=np.float32)
    T_calib = int(calib_s * FS)
    if T_calib >= T:
        raise ValueError(f"calib_s={calib_s}s exceeds signal {T/FS:.2f}s")

    # ---- Calibration via FastICA ----------------------------------
    emg_calib = emg[:, :T_calib]
    z_ext_calib_full = extension(emg_calib, ext_fact)
    D_full = z_ext_calib_full.shape[0]
    d = D_full if d_eff is None else min(d_eff, D_full - 1)
    pca = PCA(n_components=d).fit(z_ext_calib_full.T)
    z_pca_calib = pca.transform(z_ext_calib_full.T).T            # (d, T_calib)
    z_w_calib, _ = _whiten(z_pca_calib, method="ZCA")

    _src_c, spk_calib, _sil, _filters_w = run_ica_loop(
        z_w_calib, FS, sil_th=sil_th, n_iter=60,
        seed=ica_seed, refinement_loop=True, peel_off=True,
    )
    n_mu = _filters_w.shape[1]
    if n_mu == 0:
        return {}
    if verbose:
        print(f"  [yeung] calibration: {n_mu} MUs discovered in {calib_s:.1f}s")

    # ---- Initialise per-MU online state --------------------------
    min_dist = max(1, int(min_dist_ms * FS / 1000))

    mu_global = z_pca_calib.mean(axis=1)                         # μ̃z
    z_centered = z_pca_calib - mu_global[:, None]
    Sigma_global = z_centered @ z_centered.T / T_calib + cov_reg * np.eye(d)

    # Per-MU: filter v (d,), Ψ_mem FIFO, centroids (c_hi, c_lo)
    v_per = np.zeros((n_mu, d), dtype=np.float32)
    c_hi = np.zeros(n_mu, dtype=np.float32)
    c_lo = np.zeros(n_mu, dtype=np.float32)
    psi_mem: list[collections.deque] = [
        collections.deque(maxlen=psi_size) for _ in range(n_mu)
    ]

    Sigma_inv = np.linalg.inv(Sigma_global)
    for j in range(n_mu):
        sp_j = np.asarray(spk_calib.get(j, []), dtype=int)
        sp_j = sp_j[(sp_j >= 0) & (sp_j < T_calib)]
        if len(sp_j) < 5:
            continue
        # Seed Ψ_mem from calibration spike observations
        k_seed = min(psi_size, len(sp_j))
        for idx in sp_j[-k_seed:]:   # keep most-recent first
            psi_mem[j].append(z_pca_calib[:, idx].copy())
        # Eq. 9: v = mean(z̃_Ψ − μ̃z*)
        psi_arr = np.stack(list(psi_mem[j]), axis=1)
        v_per[j] = (psi_arr.mean(axis=1) - mu_global).astype(np.float32)
        # Eq. 10: c_hi = (v^T Σ^{*-1} v)^2
        c_hi[j] = float((v_per[j] @ Sigma_inv @ v_per[j]) ** 2)
        # Seed c_lo from noise amplitudes
        src = v_per[j] @ Sigma_inv @ (z_pca_calib - mu_global[:, None])
        peaks, _ = find_peaks(np.abs(src), distance=min_dist)
        if len(peaks) > 0:
            sp_set = set(int(s) for s in sp_j)
            noise_peaks = [int(p) for p in peaks if int(p) not in sp_set]
            if noise_peaks:
                noise_obs = z_pca_calib[:, noise_peaks].mean(axis=1)
                c_lo[j] = float((v_per[j] @ Sigma_inv @ (noise_obs - mu_global)) ** 2)

    # ---- Online loop ----------------------------------------------
    batch = max(1, int(batch_ms * FS / 1000))
    all_spikes: dict[int, list[int]] = {j: [] for j in range(n_mu)}
    for j in range(n_mu):
        sp_j = np.asarray(spk_calib.get(j, []), dtype=int)
        sp_j = sp_j[(sp_j >= 0) & (sp_j < T_calib)]
        all_spikes[j].extend(sp_j.tolist())

    mu_run = mu_global.copy()
    Sigma_run = Sigma_global.copy()

    s0 = T_calib
    while s0 < T:
        s1 = min(s0 + batch, T)
        if s1 - s0 < max(d, min_dist * 4):
            break

        # Window extended + PCA-projected
        emg_win = emg[:, s0:s1]
        z_win = pca.transform(extension(emg_win, ext_fact).T).T   # (d, bt)
        bt = z_win.shape[1]

        # Eq. 6 & 7: temporary EMA-updated statistics
        mu_new = z_win.mean(axis=1)
        mu_star = forgetting * mu_run + (1.0 - forgetting) * mu_new
        z_win_c = z_win - mu_star[:, None]
        Sigma_new = z_win_c @ z_win_c.T / bt
        Sigma_star = forgetting * Sigma_run + (1.0 - forgetting) * Sigma_new
        Sigma_star_inv = np.linalg.inv(Sigma_star + cov_reg * np.eye(d))

        for j in range(n_mu):
            if not np.any(v_per[j]):
                continue
            # Eq. 8: ŝ_win = v^T Σ^{*-1} (z̃ − μ*)
            src = v_per[j] @ Sigma_star_inv @ z_win_c
            # Peak detection (rectified)
            rectified = np.abs(src)
            peaks, _ = find_peaks(rectified, distance=min_dist)
            if len(peaks) < 3:
                continue
            # z-score of rectified peak amplitudes
            amp = rectified[peaks]
            mean_a, std_a = float(amp.mean()), float(amp.std() + 1e-8)
            z = (amp - mean_a) / std_a
            spike_mask = z > z_threshold
            spike_peaks = peaks[spike_mask]
            noise_peaks = peaks[~spike_mask]

            if len(spike_peaks) == 0:
                continue

            # Append new observations to Ψ_mem (FIFO auto-evicts)
            for p in spike_peaks:
                psi_mem[j].append(z_win[:, int(p)].copy())

            # Eq. 9: v ← mean(z̃_Ψ − μ*)
            psi_arr = np.stack(list(psi_mem[j]), axis=1)
            v_per[j] = (psi_arr.mean(axis=1) - mu_star).astype(np.float32)

            # Eq. 10: c_hi = (v^T Σ^{*-1} v)^2
            c_hi[j] = float((v_per[j] @ Sigma_star_inv @ v_per[j]) ** 2)
            # Eq. 11: EMA-update c_lo from average noise-peak projection
            if len(noise_peaks) > 0:
                noise_obs = z_win[:, noise_peaks].mean(axis=1)
                proj = float(v_per[j] @ Sigma_star_inv @ (noise_obs - mu_star))
                c_lo[j] = forgetting * c_lo[j] + (1.0 - forgetting) * (proj ** 2)

            # Record the same z-gated peaks as spikes (global coords)
            all_spikes[j].extend((spike_peaks + s0).tolist())

        # Commit EMA-updated global stats
        mu_run = mu_star
        Sigma_run = Sigma_star
        s0 = s1

    # ---- Deduplicate ---------------------------------------------
    min_gap = int(0.010 * FS)
    result: dict[int, np.ndarray] = {}
    for j in range(n_mu):
        sp = all_spikes[j]
        if not sp:
            continue
        sp_arr = np.unique(np.asarray(sp, dtype=int))
        if len(sp_arr) > 1:
            keep = [int(sp_arr[0])]
            for s in sp_arr[1:]:
                if s - keep[-1] >= min_gap:
                    keep.append(int(s))
            sp_arr = np.asarray(keep, dtype=int)
        if len(sp_arr) >= 5:
            result[j] = sp_arr
    return result
