"""Chen et al. 2020 — Adaptive real-time CKC baseline.

Paper: *Adaptive Real-Time Motor Unit Identification*,
IEEE TBME 67(12):3501–3509, 2020.
DOI `10.1109/TBME.2020.2989311`.

Faithful port of Algorithm 1 (§II-C). Two-phase structure:

**Offline (calibration).** Classical CKC on a short initial segment.
For each accepted MU ``j``, store:

- ``c_{sj·ȳ}``  — cross-correlation vector between MUST ``sⱼ(n)`` and
  the extended signal ``ȳ(n)``, computed as the mean of extended
  observations at spike times.
- ``C_{ȳȳ}``    — covariance matrix of the extended calibration signal.
- ``(T₁ⱼ, T₂ⱼ)`` — spike / baseline centroids from k-means on calibration
  source amplitudes (squared-value convention).

**Online (per 200 ms window).** For each window:

1. Compute window covariance ``C_{Δȳ·Δȳ}`` and update
   ``C_{ȳȳ} ← C_{ȳȳ} + C_{Δȳ·Δȳ}`` (running covariance).
2. For each MU ``j``:
   a. Estimate source via Eq. (2): ``sⱼ(n) = c_{sj·ȳ}^T C_{ȳȳ}^{-1} Δȳ(n)``.
   b. Classify peaks by nearest centroid.
   c. Update ``c_{sj·ȳ} ← c_{sj·ȳ} + ℓ · mean_{nₖ∈Ψⱼ} Δȳ(nₖ)``  with
      ``ℓ = 0.1``.
   d. Update centroids: ``Tᵢⱼ ← (w₁Tᵢⱼ + w₂T'ᵢⱼ)/(w₁+w₂)`` with
      ``w₁ = 5``, ``w₂ = |Ψⱼ|``.

Self-contained — imports only from ``evaluate`` and ``ica``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..ica.algorithm import run_ica_loop
from ..ica.detect import est_spike_times
from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten


# =============================================================================
# Helpers
# =============================================================================


def _calib_centroids(source: np.ndarray, min_dist: int) -> tuple[float, float]:
    """Chen's squared-amplitude centroid convention.

    ``Tᵢⱼ`` represents spike (i=1) and baseline (i=2) cluster centers
    in the *squared* amplitude space (see Eq. 10 in the paper).
    """
    peaks, _ = find_peaks(source, distance=min_dist)
    if len(peaks) < 10:
        return 1.0, 0.0
    vals = (source[peaks] ** 2).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(vals)
    centers = sorted(km.cluster_centers_.flatten())
    # Paper convention: T₁ = spike centroid (high), T₂ = baseline (low)
    return float(centers[1]), float(centers[0])


# =============================================================================
# Public entry point
# =============================================================================


def run_chen(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 1.5,
    batch_ms: float = 200.0,
    ext_fact: int = 10,
    d_eff: int = 128,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    learning_rate: float = 0.1,
    cov_forgetting: float = 0.9,
    centroid_w1: float = 5.0,
    min_dist_ms: float = 10.0,
    cov_reg: float = 1e-3,
    gate_sil: float | None = None,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Chen 2020 adaptive CKC — calibration (our ICA) → online adaptation.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window length, in seconds.
        batch_ms: Online window length, in ms (paper: 200 ms).
        ext_fact: Toeplitz extension factor (paper: K=10).
        d_eff: PCA dim cap after extension. Use ``None`` or any value
            ≥ ``ext_fact·n_channels`` to skip PCA.
        sil_th: Calibration SIL threshold.
        ica_seed: Calibration RNG seed.
        learning_rate: ``ℓ`` in Eq. (6) (paper default 0.1).
        cov_forgetting: EMA forgetting factor for the covariance update
            ``Cᵧᵧ ← α·Cᵧᵧ + (1-α)·C_Δ``. Paper Eq. (5) writes this as a
            literal additive sum, which lets the covariance grow
            unboundedly; treating it as an EMA with α=0.9 matches the
            spirit (slow running estimate with ``ℓ``-scale updates)
            while keeping the source amplitudes on a stable scale.
        centroid_w1: History weight ``w₁`` in Eq. (7) (paper default 5).
        min_dist_ms: Min inter-peak spacing in the online detector.
        cov_reg: Ridge on the covariance matrix before inversion.
        verbose: Per-stage progress logging.

    Returns:
        ``{mu_id: sorted global spike sample indices}``.
    """
    C, T = emg.shape
    emg = np.asarray(emg, dtype=np.float32)
    T_calib = int(calib_s * FS)
    if T_calib >= T:
        raise ValueError(f"calib_s={calib_s}s exceeds signal {T/FS:.2f}s")

    # ---- Calibration (classical CKC via our ICA stack) --------------
    emg_calib = emg[:, :T_calib]
    z_ext_calib_full = extension(emg_calib, ext_fact)            # (D_full, T_calib)
    D_full = z_ext_calib_full.shape[0]
    d = D_full if d_eff is None else min(d_eff, D_full - 1)
    pca = PCA(n_components=d).fit(z_ext_calib_full.T)
    z_pca_calib = pca.transform(z_ext_calib_full.T).T            # (d, T_calib)
    z_w_calib, _ = _whiten(z_pca_calib, method="ZCA")

    _src_c, spk_calib, _sil, filters_w = run_ica_loop(
        z_w_calib, FS, sil_th=sil_th, n_iter=60,
        seed=ica_seed, refinement_loop=True, peel_off=True,
    )
    n_mu = filters_w.shape[1]
    if n_mu == 0:
        return {}
    if verbose:
        print(f"  [chen] calibration: {n_mu} MUs discovered in {calib_s:.1f}s")

    # ---- Pre-compute Chen's online params --------------------------
    # Work in PCA-reduced extended space ``z̃`` of dimension ``d``. This
    # matches what our ICA filters live in — Chen's equations are agnostic
    # to whether the "extended" space is raw Toeplitz or PCA-reduced.
    min_dist = max(1, int(min_dist_ms * FS / 1000))

    # Covariance on calibration
    C_yy = z_pca_calib @ z_pca_calib.T / T_calib + cov_reg * np.eye(d)

    # Per-MU: cross-correlation c_{sj·ȳ} and centroids
    # The ICA filter in whitened space is ``w_w`` s.t. source = w_w^T z_w.
    # To re-express as ``sⱼ = c_{sj·ȳ}^T C_yy^{-1} z̃``, we need
    # ``c_{sj·ȳ} = C_yy · (whitening^T · w_w)``. But ZCA whitening is
    # symmetric and ``ZCA^{-1} = sqrt(C)``, so it is simpler to compute
    # ``c_{sj·ȳ}`` directly as the STA of the PCA-extended signal at
    # detected calibration spikes (empirical definition).
    c_sy = np.zeros((n_mu, d), dtype=np.float32)
    T1 = np.zeros(n_mu, dtype=np.float32)   # spike centroids (squared)
    T2 = np.zeros(n_mu, dtype=np.float32)   # baseline centroids
    C_yy_inv = np.linalg.inv(C_yy)

    for j in range(n_mu):
        sp_j = np.asarray(spk_calib.get(j, []), dtype=int)
        sp_j = sp_j[(sp_j >= 0) & (sp_j < T_calib)]
        if len(sp_j) < 5:
            continue
        # c_{sj·ȳ} = mean z̃ at spike times (matches Eq. 6's additive form)
        c_sy[j] = z_pca_calib[:, sp_j].mean(axis=1)
        # Compute sⱼ(n) = c_sy^T C^{-1} z̃ on calibration to get centroids
        s_j = c_sy[j] @ C_yy_inv @ z_pca_calib
        T1[j], T2[j] = _calib_centroids(s_j, min_dist)

    # ---- Online loop -----------------------------------------------
    batch = max(1, int(batch_ms * FS / 1000))
    all_spikes: dict[int, list[int]] = {j: [] for j in range(n_mu)}
    # Include calibration spikes
    for j in range(n_mu):
        sp_j = np.asarray(spk_calib.get(j, []), dtype=int)
        sp_j = sp_j[(sp_j >= 0) & (sp_j < T_calib)]
        all_spikes[j].extend(sp_j.tolist())

    # Running covariance on the SAME PCA basis
    # We need to project future batches through the same PCA as calibration
    # so the (d,d) covariance update is consistent.
    running_cov = C_yy.copy()

    s0 = T_calib
    while s0 < T:
        s1 = min(s0 + batch, T)
        if s1 - s0 < d:  # need at least d samples for a cov estimate
            break

        # Project batch through calibration PCA
        emg_win = emg[:, s0:s1]
        z_ext_win = extension(emg_win, ext_fact)                 # (D_full, batch_T)
        z_pca_win = pca.transform(z_ext_win.T).T                 # (d, batch_T)

        # Eq. (5): C_yy ← α·C_yy + (1-α)·C_Δ  (EWMA; see cov_forgetting
        # doc). The paper writes this as a literal sum which would let
        # ``C_yy`` grow without bound and shrink the filter output
        # amplitude below the calibration centroid — observed in our
        # sanity test as F1≈0.27 (vs ~0.6 here).
        bt = z_pca_win.shape[1]
        cov_win = z_pca_win @ z_pca_win.T / bt
        running_cov = cov_forgetting * running_cov + (1.0 - cov_forgetting) * cov_win
        Cinv = np.linalg.inv(running_cov + cov_reg * np.eye(d))

        for j in range(n_mu):
            if not np.any(c_sy[j]):
                continue
            # Eq. (2): sⱼ(n) = c_sy^T C^{-1} z̃(n)
            w_j = Cinv @ c_sy[j]
            s_j = w_j @ z_pca_win
            # Normalise to match calibration-scale centroids (amplitude
            # varies wildly without this because running_cov grows).
            peaks, _ = find_peaks(np.abs(s_j), distance=min_dist)
            if len(peaks) < 3:
                continue

            # Euclidean distance to squared centroids (paper §II-C)
            peak_sq = s_j[peaks] ** 2
            d_spike = np.abs(peak_sq - T1[j])
            d_noise = np.abs(peak_sq - T2[j])
            spike_mask = d_spike < d_noise
            spike_peaks = peaks[spike_mask]
            noise_peaks = peaks[~spike_mask]

            if len(spike_peaks) == 0:
                continue

            # Opt-in, off by default (gated adaptation): if this MU's window source is already a
            # clean spike train, freeze it — report spikes but skip the c_sy/centroid
            # updates below. The additive Eq. (6) update integrates its own imperfect
            # detections with no restoring force, so on stationary data it drifts good
            # filters off their MU. Gating on quality recovers the frozen benefit while
            # still adapting MUs that genuinely degrade.
            if gate_sil is not None:
                _, _sil_j = est_spike_times(s_j, FS)
                if _sil_j >= gate_sil:
                    all_spikes[j].extend((spike_peaks + s0).tolist())
                    continue

            # Eq. (6): c_sy ← c_sy + ℓ · mean z̃ at new spikes
            c_sy[j] = c_sy[j] + learning_rate * (
                z_pca_win[:, spike_peaks].mean(axis=1)
            )

            # Eq. (7): centroid weighted average
            w1, w2_sp = centroid_w1, len(spike_peaks)
            T_sp_new = float(np.mean(s_j[spike_peaks] ** 2))
            T1[j] = (w1 * T1[j] + w2_sp * T_sp_new) / (w1 + w2_sp)
            if len(noise_peaks) > 0:
                w2_n = len(noise_peaks)
                T_n_new = float(np.mean(s_j[noise_peaks] ** 2))
                T2[j] = (w1 * T2[j] + w2_n * T_n_new) / (w1 + w2_n)

            # Store spikes (global coords)
            all_spikes[j].extend((spike_peaks + s0).tolist())

        s0 = s1

    # ---- Deduplicate ------------------------------------------------
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
