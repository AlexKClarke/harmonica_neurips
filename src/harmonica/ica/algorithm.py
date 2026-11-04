"""FastICA decomposition loop — the package's ICA entry point.

Exposes:

- :func:`spike_triggered_average` — STA waveform from a spike train
- :func:`peel_off`                — subtract a source from the signal via STA
- :func:`run_ica_loop`            — the main fixed-point ICA loop

``run_ica_loop`` iterates: random initialisation -> fixed-point optimise ->
cluster spikes -> (optionally) CoV-ISI refine -> (optionally) peel off, then
post-processes the collection with :func:`remove_duplicates` and
:func:`remove_bad_sources`. The defaults are ``seed=1909`` and ``sil_th=0.9``;
the callers in this package pass ``sil_th`` explicitly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import fft, ifft

from .detect import est_spike_times
from .utils import gram_schmidt, remove_bad_sources, remove_duplicates


# =============================================================================
# Spike-triggered average + peel-off
# =============================================================================


def spike_triggered_average(
    sig: np.ndarray,
    spikes: np.ndarray,
    win: float = 0.02,
    fsamp: float = 2048,
) -> np.ndarray:
    """Average waveform of ``sig`` around the given spike indices.

    Args:
        sig: ``(n_channels, n_samples)`` channels-first signal.
        spikes: 1-D array of spike sample indices.
        win: Half-window size, in seconds (total window is ``2*win + 1/fsamp``
            samples).
        fsamp: Sampling frequency, in Hz.

    Returns:
        ``(n_channels, 2*width + 1)`` average waveform, where
        ``width = int(win*fsamp)``.
    """
    width = int(win * fsamp)
    waveform = np.zeros((sig.shape[0], 2 * width + 1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]
    for s in spikes:
        waveform = waveform + sig[:, (s - width) : (s + width + 1)]
    if len(spikes) > 0:
        waveform = waveform / len(spikes)
    return waveform


def peel_off(
    sig: np.ndarray,
    spikes: np.ndarray,
    win: float = 0.02,
    fsamp: float = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract a source's contribution from ``sig`` via STA-based convolution.

    Uses FFT convolution of the spike train with the STA waveform.
    The time shift introduced by FFT convolution (waveform is centred on
    ``width``) is corrected via ``numpy.roll``.

    Args:
        sig: ``(n_channels, n_samples)`` channels-first signal.
        spikes: 1-D array of spike sample indices.
        win: STA half-window, in seconds.
        fsamp: Sampling frequency, in Hz.

    Returns:
        ``(residual, component, waveform)``:

        - ``residual`` is ``sig`` with the component removed
        - ``component`` is the estimated contribution
        - ``waveform`` is the STA kernel used, ``(n_channels, 2*width+1)``
    """
    waveform = spike_triggered_average(sig, spikes, win, fsamp)

    width = int(win * fsamp)
    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]
    firings = np.zeros(sig.shape[1])
    firings[spikes] = 1

    L = sig.shape[1]
    waveform_padded = np.pad(
        waveform, ((0, 0), (0, L - waveform.shape[1])), mode="constant"
    )

    fft_firings = fft(firings)
    fft_waveform = fft(waveform_padded, axis=1)
    comp_sig = np.real(ifft(fft_waveform * fft_firings, axis=1))

    shift = (waveform.shape[1] - 1) // 2
    comp_sig = np.roll(comp_sig, -shift, axis=1)

    residual_sig = sig - comp_sig
    return residual_sig, comp_sig, waveform


# Module-level alias: the kwarg ``peel_off`` on :func:`run_ica_loop` shadows
# the function name locally, so we bind the callable to a stable name here.
_peel_off_fn = peel_off


# =============================================================================
# FastICA fixed-point decomposition loop
# =============================================================================


def run_ica_loop(
    white_sig: np.ndarray,
    fsamp: float,
    n_iter: int = 100,
    opt_max_iter: int = 100,
    opt_tol: float = 1e-4,
    opt_function_exp: float = 3,
    cluster_method: str = "kmeans",
    refinement_loop: bool = True,
    peel_off: bool = True,
    sil_th: float = 0.9,
    cov_th: float = 0.35,
    min_num_spikes: int = 10,
    match_th: float = 0.3,
    match_max_shift: float = 0.1,
    match_tol: float = 0.001,
    seed: int = 1909,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """FastICA decomposition loop with CoV-ISI refinement and peel-off.

    Each iteration draws a random initial vector, runs the fixed-point
    update with Gram-Schmidt deflation against the filters found so far,
    and detects the source's spikes.

    Args:
        white_sig: Whitened signal, ``(n_features, n_samples)``.
        fsamp: Sampling frequency, in Hz.
        n_iter: Maximum number of sources to extract.
        opt_max_iter: Maximum iterations per fixed-point optimisation.
        opt_tol: Convergence tolerance on ``|w . w_prev - 1|``.
        opt_function_exp: Contrast-function exponent (smooth ``|x|^a``).
        cluster_method: Spike clustering method. Only ``'kmeans'``.
        refinement_loop: Whether to run the CoV-ISI refinement after each
            source.
        peel_off: Whether to subtract the detected source from
            ``white_sig`` before the next iteration.
        sil_th: SIL threshold used to gate peel-off AND the final
            bad-source filter.
        cov_th: CoV-ISI threshold used to gate peel-off.
        min_num_spikes: Minimum spike count for a source to survive
            :func:`remove_bad_sources`.
        match_th: Common-spike-fraction threshold for duplicate removal.
        match_max_shift: Max shift for duplicate xcorr, in seconds.
        match_tol: Spike-matching tolerance for duplicate removal, in
            seconds.
        seed: RNG seed (``numpy.random.seed``) for the initial vectors.

    Returns:
        ``(sources, spikes, sil, mu_filters)``:

        - ``sources``   — ``(n_mu, n_samples)`` source signals
        - ``spikes``    — ``{i: spike_indices}``
        - ``sil``       — ``(n_mu,)`` SIL scores
        - ``mu_filters`` — ``(n_features, n_mu)`` extraction filters
    """
    np.random.seed(seed)
    n_features, n_samples = white_sig.shape

    # Smooth |x|^a contrast factory.
    epsilon = 1e-3

    def _make_contrast(a: float) -> tuple[Any, Any]:
        def g(x: np.ndarray) -> np.ndarray:
            return (epsilon + x ** 2) ** ((a - 3) / 2) * (a * x ** 2 + epsilon)

        def gp(x: np.ndarray) -> np.ndarray:
            return (
                (a - 1)
                * x
                * (epsilon + x ** 2) ** ((a - 5) / 2)
                * (a * x ** 2 + 3 * epsilon)
            )

        return g, gp

    sources = np.zeros((n_iter, n_samples))
    spikes: dict[int, np.ndarray] = {i: np.array([]) for i in range(n_iter)}
    sil = np.zeros(n_iter)
    mu_filters = np.zeros((n_features, n_iter))

    white_sig = white_sig.copy()

    def _run_fixed_point(
        w_init: np.ndarray, a: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """One fixed-point ICA run at exponent ``a``.

        Returns ``(w, src, sp, sil)``.
        """
        g_fp, gp_fp = _make_contrast(a)
        w = w_init.copy()
        delta = np.ones(opt_max_iter)
        k = 0
        while delta[k] > opt_tol and k < opt_max_iter - 1:
            w_last = w.copy()
            wTX = w.T @ white_sig
            A = np.mean(gp_fp(wTX))
            w = np.mean(white_sig * g_fp(wTX), axis=1) - A * w
            w = gram_schmidt(w, mu_filters)

            w = w / np.linalg.norm(w)
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

        src_out = w.T @ white_sig
        sp_out, sil_out = est_spike_times(src_out, fsamp, cluster=cluster_method)
        return w, src_out, sp_out, sil_out

    for i in range(n_iter):
        # --- initialise w ---
        w_init = np.random.randn(n_features)

        # --- fixed-point optimisation ---
        w, best_src, best_sp, best_sil_val = _run_fixed_point(
            w_init, opt_function_exp
        )

        sources[i, :] = best_src
        spikes[i] = best_sp
        sil[i] = best_sil_val

        # --- CoV-ISI refinement loop ---
        if len(spikes[i]) > 2:
            isi = np.diff(spikes[i] / fsamp)
            cov = float(np.std(isi) / np.mean(isi))
        else:
            cov = np.inf

        if len(spikes[i]) > 10 and refinement_loop:
            cov_last = cov + 1
            while cov < cov_last:
                source = w.T @ white_sig
                sp, _ = est_spike_times(source, fsamp)
                cov_last = cov
                isi = np.diff(sp / fsamp)
                cov = float(np.std(isi) / np.mean(isi))
                w = np.mean(white_sig[:, sp], axis=1)
                w = w / np.linalg.norm(w)

            sources[i, :] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(
                sources[i, :], fsamp, cluster=cluster_method
            )

        mu_filters[:, i] = w

        # --- peel-off (gated by SIL and CoV-ISI) ---
        if peel_off and sil[i] > sil_th and cov < cov_th:
            white_sig, _, _ = _peel_off_fn(
                white_sig, spikes[i], win=0.025, fsamp=fsamp
            )

    # --- post-processing ---
    sources, spikes, sil, mu_filters = remove_duplicates(
        sources, spikes, sil, mu_filters, fsamp,
        max_shift=match_max_shift, tol=match_tol, threshold=match_th,
    )
    sources, spikes, sil, mu_filters = remove_bad_sources(
        sources, spikes, sil, mu_filters,
        threshold=sil_th, min_num_spikes=min_num_spikes,
    )
    return sources, spikes, sil, mu_filters
