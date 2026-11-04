"""Mendez Guerra 2024 — Adapt baseline (wrapper around adapt_decomp package).

Paper: *Adaptive EMG decomposition in dynamic conditions based on online
learning metrics with tunable hyperparameters*, J Neural Eng 21(4), 2024.
DOI `10.1088/1741-2552/ad5ebf`.

Wraps the authors' public adapt_decomp package (the v0 snapshot in
``vendored_adapt_decomp/``; :mod:`.adapt_v1` wraps the v1.0.0 engine).
Calibration uses our own ICA stack (``ica.run_ica_loop``) with the same
Toeplitz+PCA+ZCA pipeline as the other baselines; the resulting combined
whitening matrix is projected into the delay-block space that adapt_decomp
uses internally.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .vendored_adapt_decomp.adaptation import AdaptDecomp
from .vendored_adapt_decomp.config import Config
from .vendored_adapt_decomp.preprocessing import bandpass_filter as _bandpass_filter

from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten


# =============================================================================
# Calibration helpers
# =============================================================================


def _spike_matrix(spike_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """Binary ``(n_samples,)`` vector with 1 at each spike index."""
    m = np.zeros(n_samples, dtype=np.int32)
    valid = spike_indices[(spike_indices >= 0) & (spike_indices < n_samples)]
    m[valid] = 1
    return m


def _peak_centroids(
    source: np.ndarray, min_dist: int, power: float = 2.0,
) -> tuple[float, float]:
    """Two-means spike/baseline centroids on *squared* peak amplitudes.

    AdaptDecomp detects spikes on ``|ipts|^2`` (power) and gates its IQR
    outlier filter on the same domain, so the calibration centroids must be
    in squared-amplitude space too.  This mirrors
    :func:`chen._calib_centroids`.

    ``power`` sets the domain (``|ipts|**power``); it must match the engine's
    detection exponent. v0 hard-codes 2; v1 takes ``spike_det_exp``.

    Returns ``(baseline_centroid, spike_centroid)`` in ``ipts**power`` units.
    Falls back to ``(0.0, 3.0)`` if there are fewer than 10 peaks.
    """
    peaks, _ = find_peaks(source, distance=min_dist)
    if len(peaks) < 10:
        return 0.0, 3.0
    vals = (np.abs(source[peaks]) ** power).reshape(-1, 1)   # detection domain
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(vals)
    centers = sorted(km.cluster_centers_.flatten())
    return float(centers[0]), float(centers[1])   # (baseline, spike)


def unpeeled_calib_sources(filters: np.ndarray, z_w: np.ndarray) -> np.ndarray:
    """Calibration sources recomputed on the *un-peeled* whitened signal.

    ``run_ica_loop(peel_off=True)`` returns each MU's source computed on the
    whitened signal *after* the earlier MUs were peeled off, so later MUs'
    calibration sources lack the cross-talk the online engine will see (it
    never peels). Centroids, IQR gates and contrast statistics fitted on those
    peeled sources therefore sit too low, and the online spike threshold
    admits false positives until the centroids adapt. Re-projecting the final
    filters onto the un-peeled ``z_w`` gives exactly what the online loop
    computes with the calibration whitening -- the same fix
    :func:`.chen.run_chen` applies (``s_j = c_sy^T C^{-1} z``).

    Args:
        filters: ``(d, n_mu)`` separation vectors in whitened space.
        z_w: ``(d, T_calib)`` un-peeled whitened calibration signal.

    Returns:
        ``(n_mu, T_calib)`` sources.
    """
    return filters.T @ z_w


def online_switch(T_calib: int, FS: float, online_start_s: float | None) -> int:
    """Sample at which reporting switches from calibration to online spikes.

    ``None`` (default) → ``T_calib``: the online loop starts on the batch right
    after calibration, as :func:`.chen.run_chen` / :func:`.yeung.run_yeung`
    do. ``0.0`` replays the whole recording through the online engine.
    """
    if online_start_s is None:
        return T_calib
    return min(T_calib, max(0, int(online_start_s * FS)))


def online_start(t_on: int, ext_fact: int) -> int:
    """First sample of the EMG slice handed to AdaptDecomp's online loop.

    Both adapt_decomp engines discard the first ``ext_fact`` extended rows of
    batch 0 (incomplete delay history), so starting the slice ``ext_fact``
    samples early makes the first *scored* sample exactly ``t_on``, with a
    full causal delay history. With ``t_on = T_calib`` the online loop begins
    on the batch right after calibration instead of replaying (and adapting
    on) the calibration segment.
    """
    return max(0, t_on - ext_fact)


def merge_calib_online(
    spk_calib: np.ndarray,
    spk_online: np.ndarray,
    t_on: int,
    min_gap: int,
) -> np.ndarray:
    """Calibration ICA spikes on ``[0, t_on)`` + online spikes on ``[t_on, T)``.

    Same convention as :func:`.chen.run_chen`: the calibration ICA's spikes are
    reported for the calibration window, the online tracker's afterwards, and
    the union is de-duplicated with a ``min_gap`` refractory floor so a spike
    straddling the boundary is not counted twice.

    Args:
        spk_calib: Calibration spike indices (global = calibration-local).
        spk_online: Online spike indices, already in global coordinates.
        t_on: Switch sample (see :func:`online_switch`).
        min_gap: Minimum inter-spike gap, in samples.

    Returns:
        Sorted, de-duplicated global spike indices.
    """
    spk_calib = np.asarray(spk_calib, dtype=int)
    spk_online = np.asarray(spk_online, dtype=int)
    sp = np.unique(np.concatenate([
        spk_calib[(spk_calib >= 0) & (spk_calib < t_on)],
        spk_online[spk_online >= t_on],
    ]))
    if len(sp) > 1:
        keep = [int(sp[0])]
        for s in sp[1:]:
            if s - keep[-1] >= min_gap:
                keep.append(int(s))
        sp = np.asarray(keep, dtype=int)
    return sp


# =============================================================================
# Public entry point
# =============================================================================


def run_adapt(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 2.0,
    ext_fact: int = 8,
    d_eff: int = 128,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    batch_ms: int = 100,
    adapt_wh: bool = True,
    adapt_sv: bool = True,
    adapt_sd: bool = True,
    wh_learning_rate: float = 7e-3,
    sv_learning_rate: float = 3e-3,
    device: str | None = None,
    verbose: bool = False,
    compute_loss: bool = False,
    return_loss: bool = False,
    bandpass: bool = True,
    online_start_s: float | None = None,
) -> "dict[int, np.ndarray] | tuple[dict[int, np.ndarray], float]":
    """Run Adapt (Mendez Guerra 2024) on a dynamic EMG recording.

    Calibration (first ``calib_s`` seconds) uses the same Toeplitz + PCA +
    ZCA + FastICA pipeline as the other baselines (Chen, Yeung, etc.), giving
    an ICA that operates in a well-conditioned d-dimensional space rather than
    the full D = C × ext_fact space.  The combined whitening matrix is then
    projected back to delay-block space for AdaptDecomp's online loop.

    As in Chen / Yeung, the online loop starts on the batch right after
    calibration (``T_calib``) and the calibration ICA's own spikes are
    reported on ``[0, T_calib)``; the calibration segment is not replayed
    through the adaptation.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first raw EMG.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window length, in seconds. Paper uses ``1–2 s``.
        ext_fact: Time-delay extension factor (Adapt default 10).
        d_eff: PCA dimension cap after extension (must be < C × ext_fact).
        sil_th: SIL threshold for the calibration ICA.
        ica_seed: Calibration ICA RNG seed.
        batch_ms: Online batch size in ms (Adapt default 100 ms).
        adapt_wh: Enable whitening-matrix adaptation.
        adapt_sv: Enable separation-vector adaptation.
        adapt_sd: Enable spike-detection threshold adaptation.
        wh_learning_rate: Whitening step size.
        sv_learning_rate: Separation-vector step size.
        device: ``'cuda'`` / ``'cpu'`` / ``None`` (auto).
        verbose: Print calibration progress.
        bandpass: Apply internal 20–500 Hz bandpass before processing.
            Set to ``False`` when the caller has already filtered the signal
            (e.g. 10–1000 Hz in the benchmark loop) to avoid double-filtering.
        online_start_s: Where the online loop starts, in seconds (capped at
            ``calib_s``). ``None`` (default) starts right after calibration;
            ``0.0`` replays the whole recording, including the calibration
            segment, through the online engine -- the same option
            :func:`.chen.run_chen` / :func:`.yeung.run_yeung` expose for the
            calibration-time ablation.
        compute_loss: Compute the online adaptation loss (wh + sv). Adds a
            small overhead per batch. Required for ``return_loss=True``.
        return_loss: If True, return ``(spike_dict, mean_total_loss)`` instead
            of just the spike dict.  ``compute_loss`` is automatically enabled
            when this is True.  The scalar ``mean_total_loss`` is the
            per-batch ``total_loss`` averaged over the online segment; it
            combines the whitening KL loss and the separation contrast loss,
            both normalised against calibration statistics (lower = better
            tracking of calibration conditions).

    Returns:
        ``{mu_id: sorted spike sample indices}`` when ``return_loss=False``.
        ``({mu_id: ...}, float)`` when ``return_loss=True``; the loss is
        ``None`` when calibration found no motor units.
    """
    C, T = emg.shape
    emg = np.asarray(emg, dtype=np.float32)

    # ---- Bandpass (skip if caller already filtered the signal) -----------
    # _bandpass_filter uses sosfilt(axis=-1); on (C, T) that filters along
    # the time axis — correct.
    if bandpass:
        emg_filt_cf = _bandpass_filter(
            emg,                # (C, T)
            fs=int(FS),
            cutoff=[20.0, 500.0],
            filtfilt=False,
        ).astype(np.float32)    # (C, T)
    else:
        emg_filt_cf = emg       # (C, T) — already filtered by caller

    T_calib = int(calib_s * FS)
    if T_calib >= T:
        raise ValueError(
            f"calib_s={calib_s}s exceeds signal length {T/FS:.2f}s"
        )

    # ---- Calibration: Toeplitz + PCA + ZCA + ICA -----------------------
    # Same calibration pipeline as run_chen and run_yeung (calibration window
    # only; Kramberger and Glaser whiten the whole recording).
    emg_calib = emg_filt_cf[:, :T_calib]                   # (C, T_calib)

    # 1. Toeplitz extension — same layout as other baselines
    z_ext_full = extension(emg_calib, ext_fact)             # (D_full, T_calib)
    D_full = z_ext_full.shape[0]

    # 2. PCA reduction
    d = min(d_eff, D_full - 1)
    # random_state pins sklearn's randomized SVD solver, which 'auto' selects
    # at these shapes; without it the reduction -- and so the whole calibration
    # -- is not reproducible even for a fixed ICA seed.
    pca = PCA(n_components=d, random_state=ica_seed).fit(z_ext_full.T)
    z_pca = pca.transform(z_ext_full.T).T                   # (d, T_calib)

    # 3. ZCA whitening in reduced space
    z_w, Z_zca = _whiten(z_pca, method="ZCA")              # Z_zca: (d, d)

    # 4. ICA
    src_calib, spk_calib, sil_calib, filters = run_ica_loop(
        z_w, FS,
        sil_th=sil_th,
        n_iter=60,
        seed=ica_seed,
        refinement_loop=True,
        peel_off=True,
    )
    # filters: (d, n_mu) — column per MU in whitened d-dim space
    n_mu = int(src_calib.shape[0])
    if n_mu == 0:
        # No online loop runs, so there is no loss: None (not inf/nan, which
        # mean a diverged run) lets tuning callers drop the recording rather
        # than penalise every learning-rate pair for it.
        return ({}, None) if return_loss else {}
    # ICA's src_calib is peel-off residue; use what the online loop will see.
    src_calib = unpeeled_calib_sources(filters, z_w)        # (n_mu, T_calib)
    if verbose:
        print(f"  [adapt] calibration: {n_mu} MUs in {calib_s:.1f}s "
              f"(D={D_full}→d={d})")

    # ---- Permutation P: delay-block index i*C+c → Toeplitz index c*R+i
    # x_toepl = P @ x_block  so  V_toepl @ x_toepl = V_toepl @ P @ x_block
    # (verified: P[c*ext_fact+i, i*C+c] = 1)
    P = np.zeros((D_full, D_full), dtype=np.float32)
    for c in range(C):
        for i in range(ext_fact):
            P[c * ext_fact + i, i * C + c] = 1.0

    # Combined whitening in delay-block space: (d, D_full)
    # V_block @ x_block = Z_zca @ pca.components_ @ P @ x_block
    #                   = Z_zca @ pca.transform(x_toepl)
    V_block = (Z_zca @ pca.components_) @ P                # (d, D_full)

    # ---- Build tensors for AdaptDecomp ---------------------------------
    min_spike_dist = max(1, int(0.010 * FS))  # 10 ms refractory floor

    # Online EMG — AdaptDecomp expects samples-first (T, C).  Starts ext_fact
    # samples before t_on (= T_calib by default) so the online loop's first
    # scored sample is t_on (see online_start); by default the calibration
    # segment is not replayed.
    t_on = online_switch(T_calib, FS, online_start_s)
    s_on = online_start(t_on, ext_fact)
    emg_online_t = torch.tensor(emg_filt_cf[:, s_on:].T.copy(), dtype=torch.float32)  # (T - s_on, C)
    # Calibration segment — (T_calib, C)
    emg_calib_t = torch.tensor(emg_filt_cf[:, :T_calib].T.copy(), dtype=torch.float32)  # (T_calib, C)

    # Whitening — (d, D_full) non-square
    whitening_t = torch.tensor(V_block, dtype=torch.float32)

    # Separation vectors — (n_mu, d) in whitened space
    sep_vectors_t = torch.tensor(filters.T.copy(), dtype=torch.float32)

    # IPTs — (T_calib, n_mu)
    ipts_calib_t = torch.tensor(src_calib.T.copy(), dtype=torch.float32)

    # Binary spike matrix — (T_calib, n_mu)
    sp_mat = np.stack(
        [_spike_matrix(np.asarray(spk_calib[j]), T_calib) for j in range(n_mu)],
        axis=1,
    )
    spikes_calib_t = torch.tensor(sp_mat, dtype=torch.int32)

    # Per-MU centroids in squared-amplitude space (domain matching _detect_spikes)
    base_centrs  = np.zeros(n_mu, dtype=np.float32)
    spike_centrs = np.zeros(n_mu, dtype=np.float32)
    for j in range(n_mu):
        base_centrs[j], spike_centrs[j] = _peak_centroids(
            src_calib[j], min_spike_dist
        )
    base_centr_t   = torch.tensor(base_centrs,  dtype=torch.float32)
    spikes_centr_t = torch.tensor(spike_centrs, dtype=torch.float32)

    # ---- AdaptDecomp config + run --------------------------------------
    _compute_loss = compute_loss or return_loss   # return_loss implies compute_loss

    cfg = Config(
        fs=int(FS),
        device=device,
        lowcut=20.0, highcut=500.0,
        powerline=False,
        ext_fact=ext_fact,
        batch_ms=batch_ms,
        adapt_wh=adapt_wh, adapt_sv=adapt_sv, adapt_sd=adapt_sd,
        compute_loss=_compute_loss,
        save_params=False,
        wh_learning_rate=wh_learning_rate,
        sv_learning_rate=sv_learning_rate,
        cov_alpha=0.1,
    )

    model = AdaptDecomp(
        emg=emg_online_t,
        whitening=whitening_t,
        sep_vectors=sep_vectors_t,
        base_centr=base_centr_t,
        spikes_centr=spikes_centr_t,
        emg_calib=emg_calib_t,
        ipts_calib=ipts_calib_t,
        spikes_calib=spikes_calib_t,
        preprocess=False,         # already filtered above
        config=cfg,
        store_init=False,
        save_path=None,
    )
    outputs = model.run()

    # outputs['spikes']: (T - s_on, n_mu) int32 binary tensor, indexed
    # relative to s_on.  Calibration ICA spikes cover [0, t_on).
    spikes_mat = outputs["spikes"].cpu().numpy()
    result: dict[int, np.ndarray] = {}
    for j in range(n_mu):
        idx = merge_calib_online(
            spk_calib[j],
            np.flatnonzero(spikes_mat[:, j]) + s_on,
            t_on, min_spike_dist,
        )
        if len(idx) >= 5:
            result[j] = idx

    if return_loss:
        # total_loss[b] = wh_loss[b] + sv_loss[b].nansum()  (per batch)
        # Mean over non-zero batches; nan-safe.
        loss_vec = outputs["total_loss"].cpu().numpy()
        mean_loss = float(np.nanmean(loss_vec[loss_vec != 0.0]))
        return result, mean_loss

    return result
