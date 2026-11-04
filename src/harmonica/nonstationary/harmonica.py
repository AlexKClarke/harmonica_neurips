"""Shared parts of the filter-space refiner :func:`.refractory.refine_harmonica_style`.

- :func:`gmm_cluster_mask` — 2-means spike / noise mask used by the GMM loss.
- :class:`_PerSampleNet` — Fourier-encoded MLP that outputs the per-sample
  filter correction ``dw(t)`` and bias ``b(t)``.

The paper's method (signal-space HarmonICA) is
:func:`harmonica.baselines.run_signal_space`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


# =============================================================================
# Shared utility: 2-means spike/noise mask
# =============================================================================


def gmm_cluster_mask(s_np: np.ndarray) -> np.ndarray:
    """Binary spike/noise mask from 2-means on peak amplitudes.

    Used by the GMM loss during NN training (refreshed every 15 epochs).

    Args:
        s_np: ``(T,)`` source signal.

    Returns:
        ``(T,)`` bool array, ``True`` where the sample exceeds the
        average of the two cluster centres.
    """
    peaks, _ = find_peaks(s_np, distance=20)
    if len(peaks) < 10:
        return s_np > np.percentile(s_np, 95)
    pv = s_np[peaks].reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=1, random_state=42, max_iter=50)
    km.fit(pv)
    return s_np > km.cluster_centers_.flatten().mean()


# =============================================================================
# Per-sample correction network
# =============================================================================


class _PerSampleNet(nn.Module):
    """Per-sample additive filter correction: outputs D-dim ``dw`` and scalar ``b``.

    Architecture:

    - Fourier-encode the 1-D context with frequencies ``[1, 2, 3, 5, 7, 10, 14, 20]``
    - Two-layer ReLU MLP (``hidden=64`` default)
    - Final linear projects to ``D + 1`` (``dw`` plus scalar bias)

    Initialised near zero, so refinement starts from the given filter.
    """

    def __init__(self, D_: int, n_freqs: int = 8, hidden: int = 64) -> None:
        super().__init__()
        freqs = torch.tensor(
            [1, 2, 3, 5, 7, 10, 14, 20][:n_freqs], dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D_ + 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp[-1].bias)
        self.D = D_

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = ctx.unsqueeze(-1) * self.freqs[None, :]
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        out = self.mlp(enc)
        return out[:, : self.D], out[:, self.D :]
