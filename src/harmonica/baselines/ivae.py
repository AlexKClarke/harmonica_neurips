"""iVAE nonlinear-ICA baseline — bench wrapper.

Thin ``run_ivae(emg, FS, angle=None, **kw)`` adapter around
:func:`...models.ivae.ivae_decompose` conforming to the
``run_<method>`` baseline contract used by :mod:`scripts.bench_baselines`.

Preprocessing (Toeplitz extension -> PCA -> ZCA whitening) mirrors
:func:`harmonica.nonstationary.preprocess` exactly (``R=4``,
``d_eff=128``) so iVAE sees the identical whitened input the linear-ICA
reference does.

Auxiliary variable: the joint angle, quantile-binned into pose bins and
fed as the conditioning variable ``u`` of the identifiable prior
``p(z|u)`` (the fair "iVAE-θ" variant). Every latent channel is
spike-detected with :func:`detect_best` — a deliberately generous
candidate set, so a low recovery count is a trustworthy negative rather
than an artefact of over-aggressive SIL filtering.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten
from ..nonstationary.detect import detect_best
from ..models.ivae import (
    bin_angle,
    extract_sources,
    ivae_decompose,
    make_pose_labels,
    make_segment_labels,
)


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
    """Identifiable-VAE decomposition on raw EMG.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: Optional ``(n_samples,)`` joint-angle trajectory used as the
            conditioning variable ``u`` (quantile-binned into pose bins). If
            None, falls back to equal-time segment labels.
        n_sources: Latent dimensionality — set ~ number of GT MUs for a
            fair, generous candidate set.
        n_bins: Number of angle / segment auxiliary categories.
        n_epochs: Training epochs.
        lr: Learning rate.
        encoder_hidden: Encoder hidden sizes.
        decoder_type: ``'linear'`` or ``'mlp'``.
        sparsity_weight: Kurtosis (super-Gaussianity) regulariser weight —
            10.0 is the strongest-performing prior-work setting, which
            encourages spiky sources.
        ext_fact: Toeplitz extension factor.
        d_eff: PCA dim cap after extension.
        device: Torch device.
        verbose: Print progress.

    Returns:
        ``{source_id: spike_indices}`` — spikes detected on every latent
        channel via :func:`detect_best`.
    """
    emg = np.asarray(emg, dtype=np.float32)
    T = emg.shape[1]

    # Preprocess: extension -> PCA -> ZCA (matches nonstationary.preprocess).
    z_ext = extension(emg, ext_fact)
    d = min(d_eff, z_ext.shape[0] - 1)
    z_pca = PCA(n_components=d).fit_transform(z_ext.T).T
    z_w, _ = _whiten(z_pca, method="ZCA")
    n_samp = z_w.shape[1]

    # Auxiliary variable u: quantile angle-bins (iVAE-θ) or equal-time segs.
    if angle is not None:
        pose_int = bin_angle(np.asarray(angle)[:n_samp], n_bins)
        u_labels = make_pose_labels(pose_int, n_samp)
    else:
        u_labels = make_segment_labels(n_samp, n_bins)

    # Train (ivae_decompose does its own SIL-filtered detection, but we
    # ignore that and re-detect on ALL latent channels for a generous set).
    _sc, _spc, _silc, model, _hist = ivae_decompose(
        z_w, FS, u_labels,
        n_sources=n_sources, n_epochs=n_epochs, lr=lr,
        encoder_hidden=encoder_hidden, decoder_type=decoder_type,
        sparsity_weight=sparsity_weight, sil_th=0.6,
        device=device, verbose=verbose,
    )

    all_sources = extract_sources(model, z_w, u_labels, device=device)
    result: dict[int, np.ndarray] = {}
    for i in range(all_sources.shape[0]):
        sp, _ = detect_best(all_sources[i], FS, T)
        if len(sp) >= 5:
            result[i] = sp.astype(int)
    return result
