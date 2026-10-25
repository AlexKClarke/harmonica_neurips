"""Whitening — ZCA, PCA, and Cholesky variants.

:func:`whitening` is the single entry point. ZCA with ``regularization='auto'``
is the project-wide default.
"""
from __future__ import annotations

import numpy as np


def whitening(
    Y: np.ndarray,
    method: str = "ZCA",
    backend: str = "eig",
    regularization: str | float | None = "auto",
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten a multi-channel signal.

    Args:
        Y: Input signal, ``(n_channels, n_samples)``.
        method: ``'ZCA'``, ``'PCA'``, or ``'Cholesky'``.
        backend: ``'eig'`` (Hermitian eigendecomposition of the covariance)
            or ``'svd'`` (singular-value decomposition of the data matrix).
            Ignored when ``method='Cholesky'``.
        regularization: ``'auto'`` (use the mean of the lower half of the
            eigenvalues / singular values as the shrinkage floor), a float
            (explicit shrinkage), or ``None`` (no shrinkage beyond ``eps``).
        eps: Numerical epsilon added under the square root.

    Returns:
        ``(wY, Z)`` where ``wY = Z @ Y`` is the whitened signal and ``Z``
        is the whitening matrix.
    """
    n_channels, n_samples = Y.shape

    if method == "Cholesky":
        covariance = Y @ Y.T / n_samples
        R = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(R.T)
        return Z @ Y, Z

    if backend == "svd":
        U, S, _ = np.linalg.svd(Y, full_matrices=False)
        if regularization == "auto":
            reg = float(np.mean(S[len(S) // 2 :] ** 2))
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0.0
        S_inv = 1.0 / np.sqrt(S ** 2 + reg + eps)

        if method == "ZCA":
            Z = U @ np.diag(S_inv) @ U.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ U.T
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return Z @ Y, Z

    # backend = 'eig'
    covariance = Y @ Y.T / n_samples
    S, V = np.linalg.eigh(covariance)

    if regularization == "auto":
        reg = float(np.mean(S[: len(S) // 2]))
    elif isinstance(regularization, float):
        reg = regularization
    else:
        reg = 0.0
    S_inv = 1.0 / np.sqrt(S + reg + eps)

    if method == "ZCA":
        Z = V @ np.diag(S_inv) @ V.T
    elif method == "PCA":
        Z = np.diag(S_inv) @ V.T
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return Z @ Y, Z
