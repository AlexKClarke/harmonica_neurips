"""Glaser & Holobar 2018 — Segmented cBSS baseline.

Paper: *Motor Unit Identification from High-Density Surface Electromyograms
in Repeated Dynamic Muscle Contractions*, IEEE TNSRE 27(1):66–75, 2019,
DOI `10.1109/TNSRE.2018.2885283`.

Core idea: split the dynamic recording into short (~1–2 s) near-stationary
windows, run classical CKC / cBSS on each window, then stitch spike trains
across windows by timestamp overlap. Introduces the PNR metric for quality
assessment.

This module implements the **simplified "segmented cBSS + stitch"** version
described in the paper's introduction. The paper's full *cyclostationary CKC*
with PNR-gated online tracking (Pseudocode 1, §II-C) is a deeper rewrite
left for a v2.

Re-uses only:
- :func:`harmonica.ica.algorithm.run_ica_loop` as the cBSS kernel
- :func:`harmonica.ica.whitening.whitening` (ZCA)
- :func:`harmonica.ica.utils.extension` (Toeplitz)
- :func:`harmonica.evaluate.shift_aligned_match` for cross-window MU matching
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..evaluate import shift_aligned_match
from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening


# =============================================================================
# Per-window preprocessing (independent of nonstationary/pipelines.py)
# =============================================================================


def _preprocess_window(
    emg_w: np.ndarray,
    R: int,
    d_eff: int,
) -> np.ndarray:
    """Extend → PCA → ZCA for one raw-EMG window.

    Args:
        emg_w: ``(n_channels, window_samples)`` raw EMG slice.
        R: Toeplitz extension factor.
        d_eff: PCA target dimensionality.

    Returns:
        ``(D, window_samples)`` whitened window.
    """
    z_ext = extension(emg_w, R)
    d = min(d_eff, z_ext.shape[0] - 1)
    z_pca = PCA(n_components=d).fit_transform(z_ext.T).T
    z_w, _ = whitening(z_pca, method="ZCA")
    return z_w


# =============================================================================
# Cross-window stitching
# =============================================================================


def _stitch_tracks(
    per_window_spikes: list[dict[int, np.ndarray]],
    boundaries: list[tuple[int, int]],
    FS: float,
    match_tol: float,
    match_max_shift: float,
    min_overlap_spikes: int,
    min_overlap_f1: float,
) -> list[list[tuple[int, int]]]:
    """Greedy chain-matching of MUs across adjacent windows.

    Returns a list of *tracks*, each a list of ``(window_idx, mu_idx_in_window)``
    tuples. A track represents one stitched MU.

    Matching rule: for each MU in window ``k``, find the best-F1 match to an
    unmatched track's last window's MU, scored via
    :func:`shift_aligned_match` on the **spike set within the overlap
    region**. Requires at least ``min_overlap_spikes`` spikes on each side
    and ``F1 ≥ min_overlap_f1`` to extend the track.

    Args:
        per_window_spikes: list of ``{mu_idx: global_sample_indices}`` per window.
        boundaries: list of ``(start, end)`` sample ranges per window.
        FS: Sampling frequency, in Hz.
        match_tol: Spike-matching tolerance, in seconds.
        match_max_shift: Max alignment shift for the matcher, in seconds.
        min_overlap_spikes: Minimum spikes in the overlap region on each side.
        min_overlap_f1: Minimum F1 to count as a match.
    """
    if not per_window_spikes:
        return []
    # Initialise tracks from window 0
    tracks: list[list[tuple[int, int]]] = [[(0, mu)] for mu in per_window_spikes[0]]

    for w_idx in range(1, len(per_window_spikes)):
        prev_start, prev_end = boundaries[w_idx - 1]
        cur_start, cur_end = boundaries[w_idx]
        ov_start, ov_end = max(prev_start, cur_start), min(prev_end, cur_end)

        if ov_end <= ov_start:
            # Non-overlapping — no way to stitch. Every new MU becomes a new track.
            for mu in per_window_spikes[w_idx]:
                tracks.append([(w_idx, mu)])
            continue

        # Snapshot of each track's tail (last window & MU) + its spikes in the overlap region
        tail_overlap: list[tuple[int, np.ndarray]] = []
        for t_idx, track in enumerate(tracks):
            last_w, last_mu = track[-1]
            last_sp = per_window_spikes[last_w][last_mu]
            ov_sp = last_sp[(last_sp >= ov_start) & (last_sp < ov_end)]
            tail_overlap.append((t_idx, ov_sp))

        # Each new MU's spikes in the overlap region
        new_overlap: list[tuple[int, np.ndarray]] = []
        for mu, sp in per_window_spikes[w_idx].items():
            ov_sp = sp[(sp >= ov_start) & (sp < ov_end)]
            new_overlap.append((mu, ov_sp))

        # Build score matrix: rows=new MUs, cols=existing tracks
        n_new = len(new_overlap)
        n_tracks = len(tail_overlap)
        scores = np.zeros((n_new, n_tracks))
        for ni, (_new_mu, new_sp) in enumerate(new_overlap):
            if len(new_sp) < min_overlap_spikes:
                continue
            for ti, (_t_idx, track_sp) in enumerate(tail_overlap):
                if len(track_sp) < min_overlap_spikes:
                    continue
                r = shift_aligned_match(
                    new_sp, track_sp, FS,
                    tol=match_tol, max_shift=match_max_shift,
                )
                scores[ni, ti] = r["f1"]

        # Greedy max-F1 matching (each MU / track assigned at most once)
        matched_new: set[int] = set()
        matched_track: set[int] = set()
        flat = [
            (scores[ni, ti], ni, ti)
            for ni in range(n_new) for ti in range(n_tracks)
            if scores[ni, ti] >= min_overlap_f1
        ]
        flat.sort(reverse=True)
        for _f1, ni, ti in flat:
            if ni in matched_new or ti in matched_track:
                continue
            new_mu = new_overlap[ni][0]
            tracks[ti].append((w_idx, new_mu))
            matched_new.add(ni)
            matched_track.add(ti)

        # Unmatched new MUs start new tracks
        for ni, (new_mu, _) in enumerate(new_overlap):
            if ni not in matched_new:
                tracks.append([(w_idx, new_mu)])

    return tracks


# =============================================================================
# Public entry point
# =============================================================================


def run_glaser(
    emg: np.ndarray,
    FS: float,
    window_s: float = 2.0,
    overlap_s: float = 1.0,
    R: int = 4,
    d_eff: int = 128,
    sil_th: float = 0.8,
    seed: int = 1909,
    n_iter_per_window: int = 60,
    match_tol: float = 0.002,
    match_max_shift: float = 0.025,
    min_overlap_spikes: int = 3,
    min_overlap_f1: float = 0.4,
    min_track_length: int = 1,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Segmented cBSS with cross-window spike-train stitching.

    Pipeline:

    1. Slice the raw EMG into overlapping windows of ``window_s`` seconds,
       stride ``window_s - overlap_s``.
    2. Per window: Toeplitz-extend + PCA + ZCA whiten + run fastICA
       (:func:`run_ica_loop`) to get that window's MU spike trains.
    3. Stitch MUs across adjacent windows: best-F1 match on spike trains
       within the overlap region via :func:`shift_aligned_match`.
    4. Per stitched track, concatenate spike trains and dedup within
       ``match_tol`` seconds.

    Args:
        emg: ``(n_channels, n_samples)`` raw EMG.
        FS: Sampling frequency, in Hz.
        window_s: Window length, in seconds. Glaser §III uses 1–2 s.
        overlap_s: Overlap, in seconds. ``0`` → non-overlapping (stitching
            falls back to starting new tracks per window).
        R: Toeplitz extension factor (cBSS knob, default 4).
        d_eff: PCA target dim post-extension (cBSS knob, default 128).
        sil_th: SIL threshold for :func:`run_ica_loop`.
        seed: ICA RNG seed.
        n_iter_per_window: Max ICA iterations per window.
        match_tol: Spike-matching tolerance in the stitcher, in seconds.
        match_max_shift: Max alignment shift in the stitcher, in seconds.
        min_overlap_spikes: Minimum spikes needed on each side of a window
            boundary to attempt stitching.
        min_overlap_f1: Minimum F1 (on the overlap region) to merge two MUs
            into the same stitched track.
        min_track_length: Drop tracks that span fewer than this many windows
            (defaults to 1 — keep everything).
        verbose: Per-window progress logging.

    Returns:
        ``{stitched_mu_id: sorted_global_spike_indices}`` — matches the
        other runners in :mod:`harmonica.baselines` and is directly consumable by
        :func:`harmonica.evaluate.evaluate_decomposition`.
    """
    M, T = emg.shape
    window_samples = int(window_s * FS)
    stride_samples = max(1, int((window_s - overlap_s) * FS))

    if window_samples > T:
        # Signal shorter than one window — fall back to single-window cBSS
        boundaries = [(0, T)]
    else:
        boundaries = []
        start = 0
        while start + window_samples <= T:
            boundaries.append((start, start + window_samples))
            start += stride_samples
        # Ensure we cover the tail (right-aligned final window)
        if boundaries[-1][1] < T:
            boundaries.append((T - window_samples, T))

    # Per-window cBSS
    per_window_spikes: list[dict[int, np.ndarray]] = []
    for w_idx, (s, e) in enumerate(boundaries):
        z_w = _preprocess_window(emg[:, s:e], R, d_eff)
        try:
            _, spk, _sils, _filters = run_ica_loop(
                z_w, FS, sil_th=sil_th, n_iter=n_iter_per_window,
                seed=seed, refinement_loop=True, peel_off=True,
            )
        except Exception as exc:
            if verbose:
                print(f"  [glaser] window {w_idx}/{len(boundaries)} failed: {exc}")
            per_window_spikes.append({})
            continue

        # Drop empty tracks + convert window-local indices to global
        spk_global = {
            k: (np.asarray(sp) + s).astype(int)
            for k, sp in spk.items()
            if len(sp) > 0
        }
        per_window_spikes.append(spk_global)
        if verbose:
            print(
                f"  [glaser] window {w_idx+1}/{len(boundaries)}  "
                f"[{s}:{e}]  n_mus={len(spk_global)}"
            )

    # Stitch
    tracks = _stitch_tracks(
        per_window_spikes, boundaries, FS,
        match_tol=match_tol, match_max_shift=match_max_shift,
        min_overlap_spikes=min_overlap_spikes,
        min_overlap_f1=min_overlap_f1,
    )

    # Flatten into spike dict
    result: dict[int, np.ndarray] = {}
    tol_samples = max(1, int(match_tol * FS))
    gid = 0
    for track in tracks:
        if len(track) < min_track_length:
            continue
        all_sp: list[int] = []
        for (w_idx, mu) in track:
            all_sp.extend(per_window_spikes[w_idx][mu].tolist())
        if not all_sp:
            continue
        # Dedup within tol_samples
        all_sp.sort()
        dedup = [all_sp[0]]
        for s in all_sp[1:]:
            if s - dedup[-1] > tol_samples:
                dedup.append(s)
        result[gid] = np.asarray(dedup, dtype=int)
        gid += 1

    return result
