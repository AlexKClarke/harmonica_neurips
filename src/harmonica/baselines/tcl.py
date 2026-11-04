"""Time-contrastive learning (TCL), identifiable nonlinear ICA.

``run_tcl(emg, FS, angle=None, **kw)`` follows the ``run_<method>`` contract
of :mod:`scripts.bench_baselines`. The input is whitened by
:func:`harmonica.nonstationary.pipelines.preprocess` (Toeplitz extension,
``R=4`` by default, PCA to 128, ZCA), then
:func:`harmonica.models.tcl.deep_tcl_decompose` runs on it and spikes are
detected on every source it accepts.

Auxiliary variable: the joint angle in quantile bins when ``angle`` is given
(TCL-θ), otherwise equal-time segments (TCL-t).
"""
from __future__ import annotations

import numpy as np

from ..models.tcl import angle_segment_labels, deep_tcl_decompose, segment_signal
from ..nonstationary.detect import detect_best
from ..nonstationary.pipelines import preprocess


def run_tcl(
    emg: np.ndarray,
    FS: float,
    angle: np.ndarray | None = None,
    *,
    n_segments: int = 24,
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
    """Time-contrastive learning decomposition of raw EMG.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: Optional ``(n_samples,)`` joint-angle trajectory. When given,
            the classifier is trained on quantile angle bins (TCL-θ);
            otherwise on equal-time segments (TCL-t).
        n_segments: Number of angle bins / time segments.
        hidden_dims: MLP hidden sizes (default ``[256, 128]``).
        n_epochs: Classifier training epochs.
        ext_fact: Toeplitz extension factor.
        d_eff: PCA dimension after extension.
        sil_th: SIL acceptance threshold of the ICA on the learned features.
        ica_seed: ICA RNG seed.
        ica_n_iter: Maximum number of ICA sources.
        verbose: Print training progress. This also changes how the
            classifier is trained (see :meth:`harmonica.models.tcl.DeepTCL.fit`).

    Returns:
        ``{source_id: spike_indices}`` for every accepted source with at
        least 5 spikes (:func:`detect_best`).
    """
    emg = np.asarray(emg, dtype=np.float32)
    T = emg.shape[1]
    z_w, _ = preprocess(emg, R=ext_fact, d_eff=d_eff)

    if angle is not None:
        labels = angle_segment_labels(np.asarray(angle)[:z_w.shape[1]], n_segments)
    else:
        labels = segment_signal(z_w.shape[1], n_segments)

    sources, _acc = deep_tcl_decompose(
        z_w, FS,
        n_segments=n_segments,
        hidden_dims=hidden_dims,
        n_epochs=n_epochs,
        labels=labels,
        sil_th=sil_th,
        ica_seed=ica_seed,
        ica_n_iter=ica_n_iter,
        verbose=verbose,
    )

    result: dict[int, np.ndarray] = {}
    if sources.ndim == 2 and sources.shape[0] > 0:
        for i in range(sources.shape[0]):
            sp, _ = detect_best(sources[i], FS, T)
            if len(sp) >= 5:
                result[i] = sp.astype(int)
    return result
