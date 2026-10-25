"""TCL nonlinear-ICA baseline — bench wrapper.

Thin ``run_tcl(emg, FS, angle=None, **kw)`` adapter around
:func:`...models.tcl.deep_tcl_decompose` conforming to the
``run_<method>`` baseline contract used by :mod:`scripts.bench_baselines`.

Preprocessing (Toeplitz extension -> PCA -> ZCA whitening) mirrors
:func:`harmonica.nonstationary.preprocess` exactly (``R=4``,
``d_eff=128``) so TCL sees the identical whitened input the linear-ICA
reference does — the comparison is apples-to-apples.

Auxiliary variable: the joint angle. When ``angle`` is provided the
classifier is trained on quantile angle-bin segments (the fair "TCL-θ"
variant); otherwise it falls back to equal-time segments.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..ica.utils import extension
from ..ica.whitening import whitening as _whiten
from ..nonstationary.detect import detect_best
from ..models.tcl import (
    angle_segment_labels,
    deep_tcl_decompose,
    segment_signal,
)


def run_tcl(
    emg: np.ndarray,
    FS: float,
    angle: np.ndarray | None = None,
    *,
    n_segments: int = 24,
    input_mode: str = "raw",
    hidden_dims: list[int] | None = None,
    n_epochs: int = 200,
    ext_fact: int = 4,
    d_eff: int = 128,
    sil_th: float = 0.6,
    ica_seed: int = 1909,
    ica_n_iter: int = 100,
    verbose: bool = False,
    **_ignored,
) -> dict[int, np.ndarray]:
    """Time-Contrastive Learning decomposition on raw EMG.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: Optional ``(n_samples,)`` joint-angle trajectory. When given,
            the TCL classifier is trained on quantile angle-bin segments
            (TCL-θ); otherwise equal-time segments are used.
        n_segments: Number of angle bins / time segments.
        input_mode: ``'raw'`` / ``'energy'`` / ``'both'`` (see
            :func:`deep_tcl_decompose`).
        hidden_dims: MLP hidden sizes (default ``[256, 128]``).
        n_epochs: Classifier training epochs.
        ext_fact: Toeplitz extension factor.
        d_eff: PCA dim cap after extension.
        sil_th: SIL acceptance threshold for the ICA-on-features stage.
        ica_seed: ICA RNG seed.
        ica_n_iter: Max ICA sources to extract.
        verbose: Print progress.

    Returns:
        ``{source_id: spike_indices}`` — spikes detected on each accepted
        TCL->ICA source via :func:`detect_best`.
    """
    emg = np.asarray(emg, dtype=np.float32)
    T = emg.shape[1]

    # Preprocess: extension -> PCA -> ZCA (matches nonstationary.preprocess).
    z_ext = extension(emg, ext_fact)
    d = min(d_eff, z_ext.shape[0] - 1)
    z_pca = PCA(n_components=d).fit_transform(z_ext.T).T
    z_w, _ = _whiten(z_pca, method="ZCA")

    # Auxiliary variable: angle-bin segments (TCL-θ) or equal-time fallback.
    if angle is not None:
        labels = angle_segment_labels(np.asarray(angle)[:z_w.shape[1]], n_segments)
    else:
        labels = segment_signal(z_w.shape[1], n_segments)

    sources, _spikes, _sil, _filters, _acc, _fdim = deep_tcl_decompose(
        z_w, FS,
        n_segments=n_segments,
        input_mode=input_mode,
        hidden_dims=hidden_dims,
        n_epochs=n_epochs,
        labels=labels,
        sil_th=sil_th,
        ica_seed=ica_seed,
        ica_n_iter=ica_n_iter,
        verbose=verbose,
    )

    # Detect spikes generously on every accepted source.
    result: dict[int, np.ndarray] = {}
    if sources.ndim == 2 and sources.shape[0] > 0:
        for i in range(sources.shape[0]):
            sp, _ = detect_best(sources[i], FS, T)
            if len(sp) >= 5:
                result[i] = sp.astype(int)
    return result
