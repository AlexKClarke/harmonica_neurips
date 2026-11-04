"""Kramberger & Holobar 2021 — Filter-transition baseline.

Paper: *Motor Unit Filter Changes in Electromyograms*,
IEEE Access 9:103533–103540, 2021,
DOI `10.1109/ACCESS.2021.3097793`.

Models MU filter weights as a function of a biomechanical state variable
(joint angle). Calibrates on the most stationary window of the signal,
then tracks each filter forward through the recording via a Kalman update
driven by STA measurements — *the filter IS the STA* in the CKC formulation,
so prediction and measurement share the same representation.

A faithful reimplementation of the paper's §II-B/III-B pipeline: stationary-
window CKC init → windowed forward tracking with CKC-consistent STA
measurements, then STA↔spike refinement to stay on the CKC manifold.

Requires joint-angle side information at sample resolution. Available on:

- the MUniverse benchmark recordings (``angle_profile`` in the npz)
- any npz carrying ``angle_continuous``

Not runnable on recordings without an angle trace.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..ica.algorithm import peel_off as _peel_off_fn
from ..ica.detect import est_spike_times
from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten


# =============================================================================
# Calibration helpers
# =============================================================================


def _find_most_stationary_window(
    angle: np.ndarray, fs: float, win_s: float = 1.5,
) -> tuple[int, int, float]:
    """Return the window of length ``win_s`` with the smallest angle range.

    Searches with stride ``win_samp // 4``. Returns ``(start, end, range)``.
    """
    T = len(angle)
    win_samp = int(win_s * fs)
    best_s0, best_range = 0, np.inf
    for s0 in range(0, T - win_samp, max(1, win_samp // 4)):
        s1 = s0 + win_samp
        a_range = float(angle[s0:s1].max() - angle[s0:s1].min())
        if a_range < best_range:
            best_range = a_range
            best_s0 = s0
    return best_s0, best_s0 + win_samp, best_range


def _ckc_on_window(
    z_win: np.ndarray,
    fs: float = 2048,
    n_iter: int = 30,
    sil_th: float = 0.6,
    cov_th: float = 0.5,
    max_refine: int = 30,
    min_spikes: int = 8,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """CKC decomposition on a (~stationary) whitened window.

    Activity-index initialisation + STA ↔ spike-detection refinement +
    STA-based peel-off. Relaxed thresholds since each window is short.

    Args:
        z_win: ``(D, win_samples)`` whitened window.
        fs: Sampling frequency (Hz).
        n_iter: Maximum number of sources to extract.
        sil_th: SIL acceptance threshold (relaxed 0.6 vs the global 0.9).
        cov_th: ISI coefficient-of-variation upper bound.
        max_refine: Max STA↔spike refinement iterations per source.
        min_spikes: Minimum spike count to accept a source.

    Returns:
        ``(filters, spike_dict, sils)`` where ``filters`` is ``(D, n_src)``.
    """
    D, T = z_win.shape
    z_res = z_win.copy()
    filt_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    sil_list: list[float] = []
    used = np.zeros(T, dtype=bool)

    for _ in range(n_iter):
        norms = np.sum(z_res ** 2, axis=0)
        norms[used] = 0
        n_max = int(np.argmax(norms))
        if norms[n_max] < 1e-8:
            break

        f = z_res[:, n_max].copy()
        fn = np.linalg.norm(f)
        if fn < 1e-10:
            used[max(0, n_max - 20):min(T, n_max + 20)] = True
            continue
        f /= fn

        best_sil = 0.0
        best_sp: np.ndarray = np.array([])
        best_f = f.copy()
        best_cov = np.inf

        for _ref in range(max_refine):
            s = f @ z_res
            top_idx = np.argsort(np.abs(s))[-max(5, T // 200):]
            if np.mean(s[top_idx]) < 0:
                f = -f
                s = -s

            sp, sil_val = est_spike_times(s, fs)
            if len(sp) < min_spikes:
                break

            if len(sp) > 2:
                isi = np.diff(sp / fs)
                cov = float(np.std(isi) / np.mean(isi))
            else:
                cov = np.inf

            if sil_val > best_sil:
                best_sil = sil_val
                best_sp = sp.copy()
                best_f = f.copy()
                best_cov = cov

            f_new = z_res[:, sp].mean(axis=1)
            fn = np.linalg.norm(f_new)
            if fn < 1e-10:
                break
            f_new /= fn
            if abs(float(np.dot(f, f_new))) > 0.9999:
                break
            f = f_new

        if best_sil < sil_th or len(best_sp) < min_spikes or best_cov > cov_th:
            used[max(0, n_max - 20):min(T, n_max + 20)] = True
            continue

        # Final sign check
        s_check = best_f @ z_res
        if len(best_sp) > 0 and np.mean(s_check[best_sp[:min(5, len(best_sp))]]) < 0:
            best_f = -best_f

        filt_list.append(best_f.copy())
        sp_list.append(best_sp)
        sil_list.append(best_sil)

        for p in best_sp:
            used[max(0, p - 20):min(T, p + 20)] = True

        z_res, _, _ = _peel_off_fn(z_res, best_sp, win=0.025, fsamp=fs)

    n_src = len(filt_list)
    if n_src == 0:
        return np.zeros((D, 0)), {}, np.array([])

    filters = np.column_stack(filt_list)
    spikes = {i: sp_list[i] for i in range(n_src)}
    sils = np.asarray(sil_list, dtype=float)
    return filters, spikes, sils


# =============================================================================
# Kalman-CKC tracker — filter state tracks the STA measurement
# =============================================================================


def _kramberger_track(
    z_w: np.ndarray,
    filt_init: np.ndarray,
    sp_init: dict[int, np.ndarray],
    fs: float = 2048,
    win_s: float = 1.0,
    overlap: float = 0.5,
    q: float = 1e-3,
    r_meas: float = 1e-1,
    ckc_refine: int = 3,
) -> tuple[dict[int, np.ndarray], dict[int, list]]:
    """Windowed forward tracker — filter state = CKC STA.

    Per window: predict (add process noise), apply filter + detect
    spikes, measure STA, Kalman-correct, then run a short STA↔spike
    refinement so the filter stays on the CKC manifold (filter = STA
    by definition).

    Args:
        z_w: ``(D, T)`` globally-whitened signal.
        filt_init: ``(D, n_src)`` calibration filters.
        sp_init: ``{mu: spike_indices}`` calibration spikes (for sign fix).
        fs: Sampling frequency (Hz).
        win_s: Tracking window length (s).
        overlap: Tracking window overlap fraction.
        q: Process noise variance (Kalman).
        r_meas: Measurement noise variance (Kalman).
        ckc_refine: STA↔spike refinement iterations after Kalman update.

    Returns:
        ``(spike_dict, filter_timeline)`` where ``filter_timeline[j]`` is
        a list of ``(start, end, w)`` tuples describing how the filter
        evolved over time (useful for diagnostics).
    """
    D, T = z_w.shape
    win_samp = int(win_s * fs)
    step = max(1, int(win_samp * (1 - overlap)))
    n_windows = max(1, (T - win_samp) // step + 1)
    n_src = filt_init.shape[1]

    W = filt_init.T.copy()  # (n_src, D)
    # Sign fix from calibration spikes
    for j in range(n_src):
        sp_j = np.asarray(sp_init.get(j, np.array([])))
        if len(sp_j) > 0:
            s = W[j] @ z_w
            valid = sp_j[sp_j < T]
            if len(valid) > 0 and np.mean(s[valid[:min(10, len(valid))]]) < 0:
                W[j] = -W[j]

    dW = np.zeros_like(W)
    P = np.ones((n_src, D)) * 0.1
    all_spikes: dict[int, list[int]] = {j: [] for j in range(n_src)}
    filter_timeline: dict[int, list] = {j: [] for j in range(n_src)}

    for wi in range(n_windows):
        s0 = wi * step
        s1 = min(s0 + win_samp, T)
        if s1 - s0 < win_samp // 4:
            break
        z_win = z_w[:, s0:s1]
        win_T = s1 - s0

        for j in range(n_src):
            # Predict
            if wi == 0:
                w_pred = W[j].copy()
                P_pred = P[j].copy()
            else:
                w_pred = W[j] + dW[j]
                P_pred = P[j] + q

            pn = np.linalg.norm(w_pred)
            if pn < 1e-10:
                continue
            w_pred /= pn

            # Sign fix
            s_check = w_pred @ z_win
            top = np.argsort(np.abs(s_check))[-max(5, win_T // 200):]
            if np.mean(s_check[top]) < 0:
                w_pred = -w_pred

            # Apply + detect
            s_j = w_pred @ z_win
            sp_j, _ = est_spike_times(s_j, fs)
            valid = sp_j[(sp_j >= 0) & (sp_j < win_T)]
            if len(valid) < 3:
                W[j] = w_pred
                P[j] = P_pred
                filter_timeline[j].append((s0, s1, W[j].copy()))
                continue

            # Measure: STA (= CKC filter by definition)
            w_meas = z_win[:, valid].mean(axis=1)
            mn = np.linalg.norm(w_meas)
            if mn < 1e-10:
                W[j] = w_pred
                P[j] = P_pred
                filter_timeline[j].append((s0, s1, W[j].copy()))
                continue
            w_meas /= mn

            # Kalman correct
            K_gain = P_pred / (P_pred + r_meas)
            w_corr = w_pred + K_gain * (w_meas - w_pred)
            cn = np.linalg.norm(w_corr)
            if cn < 1e-10:
                W[j] = w_pred
                filter_timeline[j].append((s0, s1, W[j].copy()))
                continue
            w_corr /= cn
            P[j] = (1 - K_gain) * P_pred

            # STA↔spike refinement — keeps filter on the CKC manifold
            f = w_corr.copy()
            for _ in range(ckc_refine):
                s_ref = f @ z_win
                top = np.argsort(np.abs(s_ref))[-max(5, win_T // 200):]
                if np.mean(s_ref[top]) < 0:
                    f = -f
                    s_ref = -s_ref
                sp_ref, _ = est_spike_times(s_ref, fs)
                sp_ref = sp_ref[(sp_ref >= 0) & (sp_ref < win_T)]
                if len(sp_ref) < 3:
                    break
                f_new = z_win[:, sp_ref].mean(axis=1)
                fn = np.linalg.norm(f_new)
                if fn < 1e-10:
                    break
                f_new /= fn
                if abs(float(np.dot(f, f_new))) > 0.999:
                    f = f_new
                    break
                f = f_new
            w_corr = f

            # Update state
            w_old = W[j].copy()
            W[j] = w_corr
            dW[j] = W[j] - w_old
            filter_timeline[j].append((s0, s1, W[j].copy()))

            # Final spike detection in the current window
            s_final = W[j] @ z_win
            top = np.argsort(np.abs(s_final))[-max(5, win_T // 200):]
            if np.mean(s_final[top]) < 0:
                s_final = -s_final
            sp_final, _ = est_spike_times(s_final, fs)
            sp_final = sp_final[(sp_final >= 0) & (sp_final < win_T)]
            all_spikes[j].extend((sp_final + s0).tolist())

    # Dedup
    result: dict[int, np.ndarray] = {}
    min_gap = int(0.01 * fs)
    for j in range(n_src):
        if not all_spikes[j]:
            continue
        sp_arr = np.unique(np.asarray(all_spikes[j], dtype=int))
        if len(sp_arr) > 1:
            keep = [int(sp_arr[0])]
            for s in sp_arr[1:]:
                if s - keep[-1] >= min_gap:
                    keep.append(int(s))
            sp_arr = np.asarray(keep, dtype=int)
        result[j] = sp_arr

    return result, filter_timeline


# =============================================================================
# Public entry point
# =============================================================================


def run_kramberger(
    emg: np.ndarray,
    FS: float,
    angle: np.ndarray | None = None,
    R: int = 4,
    d_eff: int = 128,
    calib_win_s: float = 1.5,
    calib_n_iter: int = 30,
    calib_sil_th: float = 0.6,
    calib_cov_th: float = 0.5,
    calib_min_spikes: int = 8,
    calib_window_mode: str = "angle",
    calib_start_s: float = 0.0,
    track_win_s: float = 1.0,
    track_overlap: float = 0.5,
    q: float = 1e-3,
    r_meas: float = 1e-1,
    ckc_refine: int = 3,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Kramberger-style decomposition: stationary-window CKC + Kalman tracking.

    Pipeline:

    1. Toeplitz-extend + PCA + ZCA-whiten the raw EMG.
    2. Find the most stationary angle window (smallest angle range,
       length ``calib_win_s``).
    3. Run :func:`_ckc_on_window` there → initial filters + spikes.
    4. :func:`_kramberger_track` forward through the full signal — each
       window does a Kalman prediction/correction of the filter state,
       followed by a CKC STA↔spike refinement.
    5. Return deduplicated global spike indices per MU.

    Args:
        emg: ``(n_channels, n_samples)`` raw channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: ``(n_samples,)`` joint-angle trajectory at sample resolution.
            Must be the same length as ``emg.shape[1]``.
        R: Toeplitz extension factor (default 4).
        d_eff: PCA target dim post-extension (default 128).
        calib_win_s: Calibration window length, in seconds (paper uses
            ~1-2 s; default 1.5).
        calib_n_iter: Max sources to extract in the calibration window.
        calib_sil_th: Per-source SIL acceptance threshold at calibration.
        calib_cov_th: Per-source ISI-CV upper bound at calibration.
        calib_min_spikes: Minimum spike count to accept a calibration MU.
        calib_window_mode: ``'angle'`` calibrates on the most stationary
            angle window; ``'time'`` on the window starting at
            ``calib_start_s``.
        calib_start_s: Start of the calibration window for ``'time'``, in
            seconds.
        track_win_s: Tracking window length, in seconds.
        track_overlap: Fractional overlap between tracking windows.
        q: Process-noise variance of the Kalman filter (per filter coord).
        r_meas: Measurement-noise variance of the Kalman filter.
        ckc_refine: STA↔spike refinement iterations per tracking window.
        verbose: Per-stage logging.

    Returns:
        ``{mu_id: sorted global spike sample indices}`` — directly
        consumable by :func:`harmonica.evaluate.evaluate_decomposition`.
    """
    C, T = emg.shape
    if calib_window_mode == "angle":
        if angle is None:
            raise ValueError("calib_window_mode='angle' requires angle trace")
        if len(angle) != T:
            raise ValueError(
                f"angle length {len(angle)} doesn't match emg samples {T}"
            )

    # Whiten globally (Kramberger assumes a single whitening — only the
    # filter drifts, not the covariance)
    z_ext = extension(emg, R)
    d = min(d_eff, z_ext.shape[0] - 1)
    z_pca = PCA(n_components=d).fit_transform(z_ext.T).T
    z_w, _ = _whiten(z_pca, method="ZCA")

    # Pick calibration window
    win_samp = int(calib_win_s * FS)
    if calib_window_mode == "angle":
        s0, s1, a_range = _find_most_stationary_window(angle, FS, win_s=calib_win_s)
        if verbose:
            print(
                f"  [kramberger] calib window [{s0}:{s1}]  "
                f"angle range={a_range:.3f}  ({calib_win_s:.1f}s)"
            )
    elif calib_window_mode == "time":
        s0 = int(calib_start_s * FS)
        s1 = min(s0 + win_samp, T)
        if verbose:
            print(
                f"  [kramberger] calib window [{s0}:{s1}]  "
                f"(time-based, paper-faithful, {calib_win_s:.1f}s)"
            )
    else:
        raise ValueError(
            f"calib_window_mode must be 'angle' or 'time', got {calib_window_mode!r}"
        )

    # Calibration in that window.
    z_calib = z_w[:, s0:s1]
    filters, sp_calib, sils = _ckc_on_window(
        z_calib, fs=FS,
        n_iter=calib_n_iter, sil_th=calib_sil_th, cov_th=calib_cov_th,
        min_spikes=calib_min_spikes,
    )
    n_src = filters.shape[1]
    if verbose:
        print(f"  [kramberger] calibration: {n_src} MUs discovered")
    if n_src == 0:
        return {}

    # Shift calibration spikes into global coords so the tracker can
    # use them for the first-window sign check.
    sp_calib_global = {j: (sp + s0).astype(int) for j, sp in sp_calib.items()}

    # Forward tracking
    result, _timeline = _kramberger_track(
        z_w, filters, sp_calib_global,
        fs=FS, win_s=track_win_s, overlap=track_overlap,
        q=q, r_meas=r_meas, ckc_refine=ckc_refine,
    )
    return result
