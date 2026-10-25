"""Harmonica-family refinement — per-sample NN filter correction.

Public API:

- :func:`persample_refine` — the core engine, ``w_eff(t) = w_start + dw(t)``
- :func:`iternn_refine`    — multi-pass + STA re-estimation
- :func:`gmm_w1`           — single-global-dw baseline

All three train a small MLP on Fourier-encoded context and optimise a
GMM-bimodality-based loss. These are building blocks of the filter-space
refiners in :mod:`.refractory`; the paper's method (signal-space HarmonICA)
is :func:`harmonica.baselines.run_signal_space`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

from ..ica.detect import est_spike_times
from .detect import template_spike_detect


# =============================================================================
# Shared utility: 2-means spike/noise mask
# =============================================================================


def gmm_cluster_mask(s_np: np.ndarray) -> np.ndarray:
    """Binary spike/noise mask from 2-means on peak amplitudes.

    Used by every harmonica-family loss that needs a coarse spike mask
    during NN training (refreshed every 15 epochs; see callers).

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

    Initialised near-zero so refinement starts from the ICA filter identity.
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


# =============================================================================
# GMM W=1 — single global additive correction
# =============================================================================


def gmm_w1(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    epochs: int = 150,
    max_corr: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn a time-invariant additive correction ``dw`` to an ICA filter.

    Pure GMM-bimodality loss; overparameterised ``Linear(D, D)`` navigates
    the non-convex landscape. Serves as the "no NN context" baseline for
    ablation studies.

    Args:
        z_w: ``(D, T)`` whitened signal.
        w_ica: ``(D,)`` ICA separation vector.
        epochs: Optimisation iterations.
        max_corr: Maximum ``||dw|| / ||w_ica||`` ratio.

    Returns:
        ``(source, w_eff)`` — corrected source and the final filter.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_, T = z_w.shape
    z_t = torch.tensor(z_w.T, dtype=torch.float32, device=device)
    w0 = torch.tensor(w_ica, dtype=torch.float32, device=device)
    max_a = max_corr * (w0.norm() + 1e-8)

    net = nn.Sequential(nn.Linear(D_, D_)).to(device)
    nn.init.zeros_(net[0].weight)
    nn.init.zeros_(net[0].bias)
    bias = torch.zeros(1, requires_grad=True, device=device)
    params = list(net.parameters()) + [bias]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cached_mask = None

    for ep in range(epochs):
        dw_out = net(w0)
        dw_c = dw_out * torch.clamp(max_a / (dw_out.norm() + 1e-8), max=1.0)
        s = z_t @ (w0 + dw_c) + bias

        if cached_mask is None or ep % 15 == 0:
            with torch.no_grad():
                cached_mask = gmm_cluster_mask(s.detach().cpu().numpy())
        spike_mask = torch.tensor(cached_mask, dtype=torch.bool, device=device)
        null_mask = ~spike_mask
        if spike_mask.sum() > 5 and null_mask.sum() > 5:
            loss = torch.sqrt(
                torch.var(s[spike_mask]) + torch.var(s[null_mask]) + 1e-8
            )
        else:
            s_n = s / (s.std() + 1e-8)
            loss = -torch.mean(torch.log(torch.cosh(s_n) + 1e-8))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()

    with torch.no_grad():
        dw_out = net(w0)
        dw_c = dw_out * torch.clamp(max_a / (dw_out.norm() + 1e-8), max=1.0)
        w_eff = (w0 + dw_c).detach().cpu().numpy()
        s_out = w_eff @ z_w + bias.item()
    return s_out, w_eff


# =============================================================================
# Per-sample refinement — the core engine
# =============================================================================


LossType = Literal["logcosh", "h_gmm07", "pure_gmm"]


def persample_refine(
    z_w: np.ndarray,
    w_start: np.ndarray,
    context: np.ndarray,
    loss_type: LossType = "pure_gmm",
    epochs: int = 150,
    max_corr: float = 0.4,
    seed: int | None = None,
) -> np.ndarray:
    """Per-sample additive filter correction: ``w_eff(t) = w_start + dw(t)``.

    Three loss variants:

    - ``'logcosh'``  — pure negentropy (log-cosh).
    - ``'h_gmm07'``  — ``0.3·negentropy + 0.7·GMM + smoothness + magnitude``.
    - ``'pure_gmm'`` — GMM bimodality + light smoothness (winner across
      blind benchmarks).

    Args:
        z_w: ``(D, T)`` whitened signal.
        w_start: ``(D,)`` starting filter (e.g. ICA or STA).
        context: ``(T,)`` context variable (normalised time or angle).
        loss_type: Loss-function variant.
        epochs: Training epochs.
        max_corr: ``||dw|| / ||w_start||`` budget.
        seed: Optional ``torch.manual_seed`` value.

    Returns:
        ``(T,)`` corrected source signal.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_, T = z_w.shape
    z_t = torch.tensor(z_w.T, dtype=torch.float32, device=device)
    ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
    w0 = torch.tensor(w_start, dtype=torch.float32, device=device)
    w0_norm = w0.norm() + 1e-8
    max_a = max_corr * w0_norm

    si = torch.argsort(ctx_t)
    z_s, ctx_s = z_t[si], ctx_t[si]

    model = _PerSampleNet(D_).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cached_mask = None

    model.train()
    for ep in range(epochs):
        dw_raw, b = model(ctx_s)
        dw_n = dw_raw.norm(dim=-1, keepdim=True) + 1e-8
        dw = dw_raw * torch.clamp(max_a / dw_n, max=1.0)
        w_eff = w0.unsqueeze(0) + dw
        s = (w_eff * z_s).sum(-1) + b.squeeze(-1)
        s_n = s / (s.std() + 1e-8)

        neg_ent = -torch.mean(torch.log(torch.cosh(s_n) + 1e-8))

        gmm_loss = neg_ent
        if loss_type in ("h_gmm07", "pure_gmm"):
            if cached_mask is None or ep % 15 == 0:
                with torch.no_grad():
                    cached_mask = gmm_cluster_mask(s.detach().cpu().numpy())
            spike_mask = torch.tensor(cached_mask, dtype=torch.bool, device=device)
            null_mask = ~spike_mask
            if spike_mask.sum() > 5 and null_mask.sum() > 5:
                gmm_loss = torch.sqrt(
                    torch.var(s[spike_mask]) + torch.var(s[null_mask]) + 1e-8
                )

        if loss_type == "logcosh":
            loss_main = neg_ent
        elif loss_type == "h_gmm07":
            loss_main = 0.3 * neg_ent + 0.7 * gmm_loss
        elif loss_type == "pure_gmm":
            loss_main = gmm_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

        smooth = (
            torch.mean((dw[1:] - dw[:-1]) ** 2)
            + torch.mean((b[1:] - b[:-1]) ** 2)
        )
        mag = torch.mean(dw_raw ** 2) / (w0_norm ** 2)

        loss = loss_main + 0.5 * smooth
        if loss_type == "h_gmm07":
            loss = loss_main + 1.0 * smooth + 0.5 * mag

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

    model.eval()
    with torch.no_grad():
        dw_raw, b = model(ctx_t)
        dw = dw_raw * torch.clamp(
            max_a / (dw_raw.norm(dim=-1, keepdim=True) + 1e-8), max=1.0
        )
        s_full = ((w0.unsqueeze(0) + dw) * z_t).sum(-1) + b.squeeze(-1)
    return s_full.cpu().numpy()


# =============================================================================
# Multi-pass iterNN — persample_refine + weighted-STA re-estimation
# =============================================================================


def iternn_refine(
    z_w: np.ndarray,
    w_start: np.ndarray,
    context: np.ndarray,
    n_passes: int = 3,
    mc_schedule: list[float] | None = None,
    ep_schedule: list[int] | None = None,
    loss_type: LossType = "pure_gmm",
    fsamp: float = 2048,
    seed: int | None = None,
    select: Literal["last", "sil", "max_spikes"] = "last",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Multi-pass TimeNN refinement with weighted-STA filter re-estimation.

    Each pass:

    1. Runs :func:`persample_refine` to produce a refined source.
    2. Detects spikes from the source (both k-means and template xcorr).
    3. Recomputes the filter via weighted STA (weights = ``|source|`` at
       detected spikes) — only when ``max_corr ≤ 0.4`` and ≥ 20 spikes
       detected.
    4. The updated filter seeds the next pass.

    The default ``max_corr`` schedule ``[0.3, 0.4, 0.5]`` lets the filter
    rotate progressively; the final pass is the one returned under
    ``select='last'``.

    Args:
        z_w: ``(D, T)`` whitened signal.
        w_start: ``(D,)`` initial filter (typically ICA's).
        context: ``(T,)`` context variable.
        n_passes: Number of passes (clipped to the schedule length).
        mc_schedule: Per-pass ``max_corr``. Defaults to ``[0.3, 0.4, 0.5]``.
        ep_schedule: Per-pass epochs. Defaults to ``[150, 150, 200]``.
        loss_type: Loss for each ``persample_refine`` call.
        fsamp: Sampling frequency, in Hz.
        seed: Base RNG seed. Each pass uses ``seed + i`` for determinism.
        select: Pass-selection rule. ``'last'`` (default) returns the final
            pass; ``'sil'`` is legacy (anti-correlates with F1); ``'max_spikes'``
            picks the pass with the most detections.

    Returns:
        ``(best_spikes, best_sil, best_source, w_final)``.
    """
    if mc_schedule is None:
        mc_schedule = [0.3, 0.4, 0.5]
    if ep_schedule is None:
        ep_schedule = [150, 150, 200]

    T = z_w.shape[1]
    w_curr = w_start.copy()
    candidates: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []

    for i in range(min(n_passes, len(mc_schedule))):
        mc = mc_schedule[i]
        ep = ep_schedule[i] if i < len(ep_schedule) else ep_schedule[-1]
        try:
            pass_seed = (seed + i) if seed is not None else None
            s = persample_refine(
                z_w, w_curr, context,
                loss_type=loss_type, epochs=ep, max_corr=mc, seed=pass_seed,
            )
            sp, sil = est_spike_times(s, fsamp)
            candidates.append((sp, sil, s, w_curr.copy()))

            sp_t, sil_t = template_spike_detect(s, fsamp)
            if len(sp_t) > 10:
                candidates.append((sp_t, sil_t, s, w_curr.copy()))

            sp_v = sp[(sp >= 0) & (sp < T)]
            if len(sp_v) > 20 and mc <= 0.4:
                wts = np.abs(s[sp_v.astype(int)])
                wts /= wts.sum() + 1e-10
                w_new = (z_w[:, sp_v.astype(int)] * wts[None, :]).sum(1)
                n = np.linalg.norm(w_new)
                if n > 1e-10:
                    w_curr = w_new / n
        except Exception:
            pass

    if not candidates:
        s_ica = w_start @ z_w
        sp_ica, sil_ica = est_spike_times(s_ica, fsamp)
        return sp_ica, sil_ica, s_ica, w_start

    if select == "sil":
        best = max(candidates, key=lambda c: c[1])
    elif select == "max_spikes":
        best = max(candidates, key=lambda c: len(c[0]))
    else:  # 'last'
        best = candidates[-1]
    return best[0], best[1], best[2], best[3]
