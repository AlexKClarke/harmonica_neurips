"""Identifiable VAE (iVAE), identifiable nonlinear ICA.

``run_ivae(emg, FS, angle=None, **kw)`` follows the ``run_<method>`` contract
of :mod:`scripts.bench_baselines`. The input is whitened by
:func:`harmonica.nonstationary.pipelines.preprocess` (Toeplitz extension,
``R=4`` by default, PCA to 128, ZCA), an iVAE is trained on it
(:func:`harmonica.models.ivae.ivae_decompose`), and spikes are detected on
every latent channel, a deliberately generous candidate set.

Auxiliary variable ``u`` of the prior ``p(z|u)``: the joint angle in quantile
bins when ``angle`` is given (iVAE-θ), otherwise equal-time segments. The
network is initialised from the global ``torch`` RNG; seed it beforehand for a
repeatable run.
"""
from __future__ import annotations

import numpy as np

from ..models.ivae import ivae_decompose
from ..models.tcl import angle_segment_labels, segment_signal
from ..nonstationary.detect import detect_best
from ..nonstationary.pipelines import preprocess


def run_ivae(
    emg: np.ndarray,
    FS: float,
    angle: np.ndarray | None = None,
    *,
    n_sources: int = 30,
    n_bins: int = 24,
    n_epochs: int = 300,
    lr: float = 1e-3,
    encoder_hidden: tuple[int, ...] = (256, 128),
    decoder_type: str = "linear",
    sparsity_weight: float = 10.0,
    ext_fact: int = 4,
    d_eff: int = 128,
    device: str | None = None,
    verbose: bool = False,
    **_ignored,
) -> dict[int, np.ndarray]:
    """Identifiable-VAE decomposition of raw EMG.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: Optional ``(n_samples,)`` joint-angle trajectory, binned into
            the auxiliary variable ``u``. If None, equal-time segments.
        n_sources: Latent dimensionality; the nonlinear-ICA comparison sets it
            to the number of ground-truth units.
        n_bins: Number of angle bins / time segments.
        n_epochs: Training epochs.
        lr: Learning rate.
        encoder_hidden: Encoder hidden sizes.
        decoder_type: ``'linear'`` or ``'mlp'``.
        sparsity_weight: Weight of the kurtosis (super-Gaussianity)
            regulariser, which favours spiky sources.
        ext_fact: Toeplitz extension factor.
        d_eff: PCA dimension after extension.
        device: Torch device.
        verbose: Print training progress.

    Returns:
        ``{source_id: spike_indices}`` for every latent channel with at least
        5 spikes (:func:`detect_best`).
    """
    emg = np.asarray(emg, dtype=np.float32)
    T = emg.shape[1]
    z_w, _ = preprocess(emg, R=ext_fact, d_eff=d_eff)
    n_samp = z_w.shape[1]

    if angle is not None:
        u_labels = angle_segment_labels(np.asarray(angle)[:n_samp], n_bins)
    else:
        u_labels = segment_signal(n_samp, n_bins)

    sources, _model, _hist = ivae_decompose(
        z_w, u_labels,
        n_sources=n_sources, n_epochs=n_epochs, lr=lr,
        encoder_hidden=encoder_hidden, decoder_type=decoder_type,
        sparsity_weight=sparsity_weight, device=device, verbose=verbose,
    )

    result: dict[int, np.ndarray] = {}
    for i in range(sources.shape[0]):
        sp, _ = detect_best(sources[i], FS, T)
        if len(sp) >= 5:
            result[i] = sp.astype(int)
    return result
