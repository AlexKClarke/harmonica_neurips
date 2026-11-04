"""Per-unit benchmark plumbing: preprocessing, FastICA, and P/R/F1 scoring.

Used by ``scripts/bench_sweep.py``: :func:`preprocess` whitens the
recording, :func:`run_ica` extracts the FastICA sources whose scores are
the ICA / cBSS reference row, and :func:`_prf` scores a refined unit.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..evaluate import evaluate_decomposition
from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess(
    emg: np.ndarray,
    R: int = 4,
    d_eff: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw EMG → whitened signal ready for ICA.

    Steps:

    1. Toeplitz extension with factor ``R``
    2. PCA to ``d_eff`` components (capped at ``D - 1``)
    3. ZCA whitening with auto regularisation

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG.
        R: Toeplitz extension factor.
        d_eff: PCA target dimensionality (the project default is 128).

    Returns:
        ``(z_w, z_ext)`` — whitened signal and the extended-but-not-yet-
        whitened signal.
    """
    z_ext = extension(emg, R)
    d = min(d_eff, z_ext.shape[0] - 1)
    # random_state pinned: at these sizes sklearn picks the randomized SVD
    # solver, which is not repeatable without a seed.
    z_pca = PCA(n_components=d, random_state=0).fit_transform(z_ext.T).T
    z_w, _ = whitening(z_pca, method="ZCA")
    return z_w, z_ext


def run_ica(
    z_w: np.ndarray,
    FS: float,
    n_mu: int,
    seed: int = 1909,
    sil_th: float = 0.8,
    peel_off: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """FastICA wrapper — caps iterations at ``min(3*n_mu, 108)`` and returns
    spike dict, SILs, filters, and a normalised time-context vector.

    Args:
        z_w: ``(D, T)`` whitened signal.
        FS: Sampling frequency, in Hz.
        n_mu: Expected number of MUs (used to size ``n_iter``).
        seed: ICA RNG seed.
        sil_th: SIL threshold passed to :func:`run_ica_loop`.
        peel_off: Whether to enable :func:`run_ica_loop`'s built-in
            STA peel-off between sources (default ``False``).

    Returns:
        ``(spk, sils, filters, t_norm)`` where

        - ``spk`` is ``{est_id: spike_indices}``
        - ``sils`` are SIL scores
        - ``filters`` is ``(D, n_sources)``
        - ``t_norm`` is a ``(T,)`` normalised time vector in ``[-1, 1]``
    """
    n_iter = min(n_mu * 3, 108)
    _, spk, sils, filters = run_ica_loop(
        z_w, FS, sil_th=sil_th, n_iter=n_iter,
        seed=seed, refinement_loop=True, peel_off=peel_off,
    )
    T = z_w.shape[1]
    t_norm = np.linspace(-1, 1, T).astype(np.float32)
    return spk, sils, filters, t_norm


# =============================================================================
# Scoring
# =============================================================================


def _prf(
    sp: np.ndarray | None, gt_sp: np.ndarray, FS: float
) -> tuple[float, float, float]:
    """Shift-aligned (precision, recall, F1) for a spike array vs GT."""
    if sp is None or len(sp) < 5:
        return 0.0, 0.0, 0.0
    ev = evaluate_decomposition({0: sp}, {0: gt_sp}, FS)
    if not ev["per_source"]:
        return 0.0, 0.0, 0.0
    ps = ev["per_source"][0]
    return ps["precision"], ps["recall"], ps["f1"]
