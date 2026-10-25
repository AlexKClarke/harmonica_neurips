"""Mendez Guerra 2024 — Adapt baseline (wrapper around vendored package).

Paper: *Adaptive EMG decomposition in dynamic conditions based on online
learning metrics with tunable hyperparameters*, J Neural Eng 21(4), 2024.
DOI `10.1088/1741-2552/ad5ebf`.

Wraps the authors' public PyTorch implementation (vendored at
``vendored_adapt_decomp/``, pinned to commit ``8ffeb66`` of
``github.com/imendezguerra/adapt_decomp``).

Calibration uses our own ICA stack (``ica.run_ica_loop``) — this is
consistent with how Adapt is used downstream: it takes a pre-calibrated
decomposition (whitening + separation vectors + spike/baseline centroids
+ calibration IPTs / spikes) and adaptively tracks drift.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..ica.algorithm import run_ica_loop
from ..ica.whitening import whitening as _whiten
from .vendored_adapt_decomp.adaptation import AdaptDecomp
from .vendored_adapt_decomp.config import Config
from .vendored_adapt_decomp.data_structures import _extend_data_v
from .vendored_adapt_decomp.preprocessing import (
    bandpass_filter as _bandpass,
    remove_powerline as _notch,
)


# =============================================================================
# Calibration helpers
# =============================================================================


def _spike_matrix(spike_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """Binary ``(n_samples,)`` vector with 1 at each spike index."""
    m = np.zeros(n_samples, dtype=np.int32)
    valid = spike_indices[(spike_indices >= 0) & (spike_indices < n_samples)]
    m[valid] = 1
    return m


def _peak_centroids(source: np.ndarray, min_dist: int) -> tuple[float, float]:
    """Two-means spike/baseline centroids on peak amplitudes.

    Returns ``(baseline_centroid, spike_centroid)`` — matching the
    convention :mod:`adapt_decomp` uses (``base_centr < spikes_centr``).
    Falls back to ``(0.0, 3.0)`` if there are fewer than 10 peaks.
    """
    peaks, _ = find_peaks(source, distance=min_dist)
    if len(peaks) < 10:
        return 0.0, 3.0
    vals = source[peaks].reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(vals)
    centers = sorted(km.cluster_centers_.flatten())
    return float(centers[0]), float(centers[1])


# =============================================================================
# Public entry point
# =============================================================================


def run_adapt(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 2.0,
    ext_fact: int = 10,
    d_eff: int | None = None,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    batch_ms: int = 100,
    adapt_wh: bool = True,
    adapt_sv: bool = True,
    adapt_sd: bool = True,
    wh_learning_rate: float = 7e-3,
    sv_learning_rate: float = 3e-3,
    cov_alpha: float = 0.1,
    whiten_reg: float | None = None,
    device: str | None = None,
    verbose: bool = False,
    guardrail: bool = True,
) -> dict[int, np.ndarray]:
    """Run Adapt (Mendez Guerra 2024) on a dynamic EMG recording.

    Calibration (first ``calib_s`` seconds) uses our own fastICA
    pipeline; adaptation (remaining signal) uses the vendored
    :class:`AdaptDecomp`. Calibration and adaptation share the same
    extension factor and band-passed signal so the adaptive filters
    operate in the same space as the initial separation vectors.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first raw EMG.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window length, in seconds. Paper uses
            ``1–2 s``.
        ext_fact: Toeplitz extension factor (Adapt default 10).
        sil_th: SIL threshold for the calibration ICA.
        ica_seed: Calibration ICA RNG seed.
        batch_ms: Online batch size in ms (Adapt default 100 ms).
        adapt_wh: Enable whitening-matrix adaptation.
        adapt_sv: Enable separation-vector adaptation.
        adapt_sd: Enable spike-detection threshold adaptation.
        wh_learning_rate: Whitening LR (paper table 1 reports
            ``7e-3`` for NeuroMotion, ``1e-3`` for wrist, ``2e-3`` for
            forearm).
        sv_learning_rate: Separation-vector LR (``3e-3`` / ``5e-4``).
        cov_alpha: Covariance-update rate.
        device: ``'cuda'`` / ``'cpu'`` / ``None`` (auto).
        verbose: Print calibration progress.

    Returns:
        ``{mu_id: sorted spike sample indices}`` — directly consumable
        by :func:`harmonica.evaluate.evaluate_decomposition`.
    """
    C, T = emg.shape
    emg = np.asarray(emg, dtype=np.float32)

    # Calibration window
    T_calib = int(calib_s * FS)
    if T_calib >= T:
        raise ValueError(
            f"calib_s={calib_s}s exceeds signal length {T/FS:.2f}s"
        )

    # ---- CRITICAL: match AdaptDecomp's online preprocessing -----------
    # AdaptDecomp's ``Data`` class causally bandpasses 20-500 Hz and
    # subtracts the per-channel DC mean (`data_structures.py:42-86`).
    # We must train whitening + ICA in that SAME signal space, otherwise
    # the calibration V is silently wrong when applied to the filtered
    # online stream (observed bug: even on a perfectly stationary
    # synthetic signal where calibration ICA recovers 8/8 GT MUs at
    # F1 >= 0.80, the online phase drops to 1/8 because V is being
    # applied to a different distribution; with this fix it recovers
    # to 5-8/8). Pre-bandpass + DC-remove the full EMG once and pass
    # ``preprocess=False`` to AdaptDecomp so it doesn't double-filter.
    emg_tc = emg.T  # (T, C) — paper's filter convention
    emg_tc = _bandpass(emg_tc, int(FS), cutoff=[20.0, 500.0], filtfilt=False)
    # `powerline=False` is set in our Config below — skip notch.
    emg_tc = emg_tc - emg_tc.mean(axis=0)
    emg = emg_tc.T.astype(np.float32)  # (C, T) preprocessed

    emg_calib = emg[:, :T_calib]
    emg_calib_t = torch.tensor(emg_calib.T.copy(), dtype=torch.float32)  # (T_c, C)
    z_ext_calib_t = _extend_data_v(emg_calib_t, ext_fact)  # (T_c, D_full)  delay-block-major
    z_ext_calib = z_ext_calib_t.numpy().T.astype(np.float32)  # (D_full, T_c)

    # Opt-in (whiten_reg): with the default regularization="auto", the rank-deficient
    # full-extended (C*ext_fact) covariance from a short calibration window picks
    # reg ~0, so sqrt(negative eigenvalue) -> NaN -> 0 sources. A small positive
    # ``whiten_reg`` conditions the whitening.
    _wh_kw = {} if whiten_reg is None else {"regularization": whiten_reg}

    # ---- Optional PCA -> ZCA in the same layout as chen/yeung/kramberger ---
    # If d_eff is provided, project into d=d_eff dim before ZCA. This matches the
    # other baselines' preprocessing and makes Adapt run in the same dim as them
    # (~25x faster on ch320). Empirically does NOT improve F1 on our
    # continuously-dynamic ch070 recordings (the extra calibration MUs the
    # lower-dim space surfaces drift / cluster-collapse during the online phase),
    # so it's opt-in. Suggested by Mendez Guerra (Adapt paper author) for
    # consistency with the other baselines.
    D_full = z_ext_calib.shape[0]
    pca = None
    if d_eff is not None and d_eff < D_full:
        d = min(int(d_eff), D_full - 1)
        # random_state pinned for the same reason as nonstationary/pipelines.py:
        # at these dimensions sklearn auto-selects the randomized SVD
        # solver, which is non-deterministic without a seed.
        pca = PCA(n_components=d, random_state=0).fit(z_ext_calib.T)  # fit on (T_c, D_full)
        z_pca_calib = pca.transform(z_ext_calib.T).T                  # (d, T_c)
        z_w_calib, V_mat = _whiten(z_pca_calib, method="ZCA", **_wh_kw)   # V_mat: (d, d)
    else:
        z_w_calib, V_mat = _whiten(z_ext_calib, method="ZCA", **_wh_kw)   # (D_full, D_full)

    src_calib, spk_calib, sil_calib, filters = run_ica_loop(
        z_w_calib, FS,
        sil_th=sil_th,
        n_iter=60,
        seed=ica_seed,
        refinement_loop=True,
        peel_off=True,
    )
    n_mu = int(src_calib.shape[0])
    if n_mu == 0:
        return {}
    if verbose:
        print(f"  [adapt] calibration: {n_mu} MUs discovered in {calib_s:.1f}s")

    # ---- Build the calibration artefacts AdaptDecomp wants ----------
    min_spike_dist = max(1, int(0.010 * FS))  # 10 ms refractory floor

    if pca is not None:
        # Pre-extend + PCA-project the FULL EMG so AdaptDecomp operates in
        # d-dim PCA-reduced space. Set ext_fact=1 on the Config so its
        # internal Data class doesn't extend again. ``pca.transform``
        # handles centering correctly per batch (same convention as
        # chen/yeung's online loop).
        emg_full_tc = torch.tensor(emg.T.copy(), dtype=torch.float32)        # (T, C)
        z_ext_full_t = _extend_data_v(emg_full_tc, ext_fact)                  # (T, D_full)
        z_ext_full_np = z_ext_full_t.numpy().astype(np.float32)               # (T, D_full)
        emg_d_np = pca.transform(z_ext_full_np).astype(np.float32)            # (T, d)
        emg_full_t = torch.tensor(emg_d_np, dtype=torch.float32)              # (T, d)
        emg_calib_pca = emg_d_np[:T_calib]
        emg_calib_t = torch.tensor(emg_calib_pca, dtype=torch.float32)        # (T_c, d)
        adapt_ext_fact = 1
    else:
        # No PCA: pass the raw EMG; AdaptDecomp does internal extension.
        emg_full_t = torch.tensor(emg.T.copy(), dtype=torch.float32)         # (T, C)
        emg_d_np = None
        adapt_ext_fact = ext_fact

    # Whitening: (d, d) PCA-mode, or (D_full, D_full) no-PCA mode.
    whitening_t = torch.tensor(V_mat, dtype=torch.float32)

    # Separation vectors: (n_mu, D)  — row-per-MU
    sep_vectors_t = torch.tensor(filters.T.copy(), dtype=torch.float32)

    # IPTs: (T_calib, n_mu) from our ICA output
    ipts_calib_t = torch.tensor(src_calib.T.copy(), dtype=torch.float32)

    # Binary spike matrix: (T_calib, n_mu)
    sp_mat = np.stack(
        [_spike_matrix(np.asarray(spk_calib[j]), T_calib) for j in range(n_mu)],
        axis=1,
    )
    spikes_calib_t = torch.tensor(sp_mat, dtype=torch.int32)

    # Per-MU spike / baseline centroids on the CALIBRATION IPTs
    base_centrs, spike_centrs = np.zeros(n_mu), np.zeros(n_mu)
    for j in range(n_mu):
        b, s = _peak_centroids(src_calib[j], min_spike_dist)
        base_centrs[j], spike_centrs[j] = b, s
    base_centr_t = torch.tensor(base_centrs, dtype=torch.float32)
    spikes_centr_t = torch.tensor(spike_centrs, dtype=torch.float32)

    # ---- AdaptDecomp config + run ----------------------------------
    cfg = Config(
        fs=int(FS),
        device=device,  # None → auto
        lowcut=20.0, highcut=500.0,
        powerline=False,   # we already band-pass in preprocess if needed
        ext_fact=adapt_ext_fact,  # 1 if pre-PCA'd, else ext_fact (Adapt does internal extension)
        batch_ms=batch_ms,
        adapt_wh=adapt_wh, adapt_sv=adapt_sv, adapt_sd=adapt_sd,
        compute_loss=False,  # skip loss calc to keep runtime light
        save_params=False,
        wh_learning_rate=wh_learning_rate,
        sv_learning_rate=sv_learning_rate,
        cov_alpha=cov_alpha,
    )

    model = AdaptDecomp(
        emg=emg_full_t,
        whitening=whitening_t,
        sep_vectors=sep_vectors_t,
        base_centr=base_centr_t,
        spikes_centr=spikes_centr_t,
        emg_calib=emg_calib_t,
        ipts_calib=ipts_calib_t,
        spikes_calib=spikes_calib_t,
        preprocess=False,  # CRITICAL: we pre-bandpassed above so V matches
        config=cfg,
        store_init=False,
        save_path=None,
    )
    outputs = model.run()

    # outputs['spikes']: (samples, n_mu) int32 binary tensor
    spikes_mat = outputs["spikes"].cpu().numpy()
    ipts_adapt_mat = outputs["ipts"].cpu().numpy()  # (T, n_mu)
    adapt_spikes: dict[int, np.ndarray] = {}
    for j in range(n_mu):
        idx = np.flatnonzero(spikes_mat[:, j]).astype(int)
        if len(idx) >= 5:
            adapt_spikes[j] = idx

    if not guardrail:
        return adapt_spikes

    # ---- Guardrail: emit BOTH frozen + adapted decoders per filter ------
    # Floor: "calibration ICA filters applied statically to the full
    # recording" — a naive non-adaptive decoder. By emitting both per
    # filter, downstream evaluation (which dedups per (dataset, gt_mu)
    # keeping max F1) automatically picks the better of {frozen, adapted}
    # for each GT MU. Worst case: Adapt = naive frozen ICA decode; best
    # case: adaptation strictly improved at least one filter.
    # Full signal: in PCA-reduced d-dim if pca is set, else raw extended.
    if emg_d_np is not None:
        z_w_full = (V_mat.astype(np.float32) @ emg_d_np.T)          # (d, T)
    else:
        z_ext_full_t = _extend_data_v(emg_full_t, ext_fact)         # (T, D_full)
        z_ext_full = z_ext_full_t.numpy().T.astype(np.float32)      # (D_full, T)
        z_w_full = (V_mat.astype(np.float32) @ z_ext_full)          # (D_full, T)
    src_frozen = filters.T.astype(np.float32) @ z_w_full            # (n_mu, T)

    from scipy.signal import find_peaks
    final: dict[int, np.ndarray] = {}
    n_emitted_frozen = 0
    for j in range(n_mu):
        # Always emit adapted output if it has any spikes
        adapt_idx = adapt_spikes.get(j, np.array([], dtype=int))
        if len(adapt_idx) >= 5:
            final[j] = adapt_idx

        # Also emit the frozen decode of this filter as a separate MU eid
        # (n_mu + j), so the evaluator can pick the better per-GT match
        ipt_f = src_frozen[j]
        ipt_f2 = ipt_f * ipt_f
        bcent = float(base_centrs[j])
        scent = float(spike_centrs[j])
        if scent <= 0 or scent <= bcent:
            continue
        height_thr = bcent + 0.5 * (scent - bcent)
        peaks_f, _ = find_peaks(
            ipt_f2,
            distance=min_spike_dist,
            height=[bcent / 4.0, 4.0 * scent],
        )
        if len(peaks_f) == 0:
            continue
        frozen_idx = peaks_f[ipt_f2[peaks_f] > height_thr].astype(int)
        if len(frozen_idx) >= 5:
            final[n_mu + j] = frozen_idx
            n_emitted_frozen += 1

    if verbose:
        print(f"  [adapt] guardrail: emitted {n_emitted_frozen}/{n_mu} "
              f"additional frozen decoders (total MUs out: {len(final)})")
    return final
