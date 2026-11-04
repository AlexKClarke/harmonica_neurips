"""Filter-space refiner ``refine_harmonica_style`` and its loss terms.

``refine_harmonica_style`` is the filter-space counterpart of HarmonICA
(additive ``w + dw(t)`` instead of a signal-space transform); the paper
uses it for the stationarity-loss ablation in the supplementary material.

Besides the stationarity loss, the compensation step penalises peaks in a
refractory violation zone around the anchor spikes:

- **Dead zone** (``|Δt| < min_isi``) — peaks are ignored; the same unit
  cannot fire again this soon.
- **Violation zone** (``min_isi ≤ |Δt| ≤ max_suppress_factor × median_ISI``)
  — peaks are penalised during training, with a triangular weight that is
  1 at the refractory floor and decays to 0 at ``max_suppress``. Another
  unit's interference leaks into this range.
- **Free zone** (beyond) — not penalised; a peak here may be a real firing
  the ICA source missed.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks

from .harmonica import _PerSampleNet, gmm_cluster_mask

# =============================================================================
# Compensation losses (supplementary loss ablation)
# =============================================================================

COMP_LOSSES = ("gmm", "negentropy_g1", "negentropy_g2", "kurtosis", "power_p")


def _comp_loss(
    s: torch.Tensor,
    spike_mask: torch.Tensor,
    null_mask: torch.Tensor,
    kind: str = "gmm",
    nonlin_scale: float = 10.0,
    sep_exponent: float = 4.0,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """Stationarity-loss family compared in the supplementary loss ablation.

    All variants are returned as a quantity to **minimize**.

    - ``gmm`` — bimodality via 2-cluster intra-variance (paper default,
      with FastICA-G1 fallback when the cluster split is degenerate).
    - ``negentropy_g1`` — ``E[log cosh(s_n)]`` (FastICA G1, super-Gaussian).
    - ``negentropy_g2`` — ``E[exp(-s_n^2/2)]`` (FastICA G2, robust contrast).
    - ``kurtosis`` — ``-E[s_n^4]`` (super-Gaussian / sparse).
    - ``power_p`` — HarmonICA-engine sign-preserving power contrast on
      ``nonlin_scale * s_n``, exponent ``sep_exponent``.
    """
    s_n = (s - s.mean()) / (s.std() + 1e-8)
    if kind == "gmm":
        if spike_mask.sum() > 5 and null_mask.sum() > 5:
            return torch.sqrt(
                torch.var(s[spike_mask]) + torch.var(s[null_mask]) + 1e-8
            )
        return -torch.mean(torch.log(torch.cosh(s_n) + 1e-8))
    if kind == "negentropy_g1":
        return -torch.mean(torch.log(torch.cosh(s_n) + 1e-8))
    if kind == "negentropy_g2":
        return torch.mean(torch.exp(-0.5 * s_n.pow(2).clamp(max=50.0)))
    if kind == "kurtosis":
        return -torch.mean(s_n.pow(4))
    if kind == "power_p":
        s_c = s_n.clamp(max=clamp_max)
        return -torch.mean(s_c.sign() * (nonlin_scale * s_c).abs().pow(sep_exponent))
    raise ValueError(f"unknown comp_loss: {kind!r} (allowed: {COMP_LOSSES})")


# =============================================================================
# Violation detection with triangular distance weights
# =============================================================================


def _find_refractory_violations(
    s_np: np.ndarray,
    anchor_spikes: np.ndarray,
    FS: float,
    min_isi_ms: float = 30.0,
    max_suppress_factor: float = 0.5,
    return_weights: bool = False,
):
    """Return peaks that fall in the refractory violation zone around anchors.

    A peak at time ``t`` is a violation if there exists an anchor ``t_a``
    with ``min_isi ≤ |t - t_a| ≤ max_suppress``, where ``max_suppress =
    max_suppress_factor × median(Δanchor)``.

    Args:
        s_np: ``(T,)`` source signal.
        anchor_spikes: Accepted spike sample indices.
        FS: Sampling frequency, in Hz.
        min_isi_ms: Refractory floor, in ms.
        max_suppress_factor: Fraction of the median anchor ISI to count as
            the violation-zone upper bound.
        return_weights: If True, also return triangular distance weights —
            ``w = 1`` at ``min_isi``, decaying linearly to ``w = 0`` at
            ``max_suppress``.

    Returns:
        Either ``violations`` or ``(violations, weights)`` when
        ``return_weights=True``. ``violations`` is an int array of sample
        indices.
    """
    anchors = np.sort(anchor_spikes.astype(int))
    anchors = anchors[(anchors >= 0) & (anchors < len(s_np))]
    if len(anchors) < 2:
        empty = np.array([], int)
        return (empty, np.array([], float)) if return_weights else empty

    expected_isi = np.median(np.diff(anchors))
    min_isi = int(min_isi_ms / 1000 * FS)
    max_suppress = int(expected_isi * max_suppress_factor)

    peaks, _ = find_peaks(s_np, distance=max(1, int(FS * 0.005)))
    anchor_set = set(anchors.tolist())
    violations: list[int] = []
    weights: list[float] = []
    for pk in peaks:
        if pk in anchor_set:
            continue
        nearest_dist = int(np.min(np.abs(anchors - pk)))
        if min_isi <= nearest_dist <= max_suppress:
            violations.append(int(pk))
            if return_weights:
                w = 1.0 - (nearest_dist - min_isi) / max(1, max_suppress - min_isi)
                weights.append(max(0.0, min(1.0, w)))
    viol_arr = np.array(violations, int)
    if return_weights:
        return viol_arr, np.array(weights, float)
    return viol_arr


# =============================================================================
# Alternating W + NN compensation (HarmonICA-style)
# =============================================================================


def refine_harmonica_style(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    anchor_spikes: np.ndarray,
    epochs: int = 150,
    sep_lr: float = 1e-3,
    comp_lr: float = 5e-4,
    sep_exponent: float = 4.0,
    nonlin_scale: float = 10.0,
    clamp_max: float = 10.0,
    max_corr: float = 0.5,
    lam_refract: float = 2.0,
    distance_weighted: bool = True,
    refract_exponent: int = 2,
    seed: int | None = 42,
    compensation_delay: int = 0,
    comp_loss: str = "gmm",
) -> np.ndarray:
    """HarmonICA-style alternating refinement in filter space (W + NN compensation).

    Applies HarmonICA's alternating scheme to the filter rather than the
    signal, warm-started from ``w_ica`` (no projection pursuit):

    - The separation weight ``W`` is trainable, with its own optimiser.
      Its loss is the sign-preserving power contrast
      ``L_sep = -E[sign(s_raw) · |nonlin_scale · s_raw|^sep_exponent]`` on
      the *uncompensated* source ``s_raw = W^T z_w``, the engine's contrast
      with different defaults (``sep_exponent=4``, ``nonlin_scale=10``,
      ``clamp_max=10``).
    - The compensation network outputs ``dw(t), b(t)``. Its loss is the
      stationarity loss ``comp_loss`` + weighted refractory suppression +
      smoothness, on the compensated source
      ``s = (W + dw(t))^T z_w + b(t)`` with ``W`` detached.
    - The two optimisers alternate once per epoch, like the engine's
      ``joint_step``.

    Args:
        z_w: ``(D, T)`` whitened signal.
        w_ica: ``(D,)`` starting separation weight (a FastICA filter).
        t_norm: ``(T,)`` normalised time context.
        FS: Sampling frequency, in Hz.
        anchor_spikes: Accepted spike indices (for refractory suppression).
        epochs: Total alternating steps.
        sep_lr: Learning rate for the separation weight.
        comp_lr: Learning rate for the compensation NN.
        sep_exponent: Exponent in the sign-preserving separation loss.
        nonlin_scale: Scale factor inside the separation loss.
        clamp_max: Clamp for the normalised ``s_raw`` before the power.
        max_corr: ``||dw|| / ||w||`` budget.
        lam_refract: Refractory-suppression loss weight.
        distance_weighted: Use triangular violation weights.
        refract_exponent: Exponent in the refractory loss (2 quadratic,
            4 stronger).
        seed: Optional ``torch.manual_seed``.
        compensation_delay: Number of separation-only warm-up steps before
            the compensation NN joins (default 0: active from epoch 0).
        comp_loss: Stationarity loss of the compensation step, one of
            :data:`COMP_LOSSES`. ``"gmm"`` (default) is the GMM bimodality
            loss; the others are the classical contrasts compared in the
            supplementary loss ablation.

    Returns:
        ``(T,)`` refined compensated source signal (original time order).
    """
    import torch as _torch

    if seed is not None:
        _torch.manual_seed(seed)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    D_, T = z_w.shape
    z_t = _torch.tensor(z_w.T, dtype=_torch.float32, device=device)
    ctx_t = _torch.tensor(t_norm, dtype=_torch.float32, device=device)

    # Trainable separation weight, initialised at w_ica
    w = _torch.nn.Parameter(
        _torch.tensor(w_ica, dtype=_torch.float32, device=device)
    )

    si = _torch.argsort(ctx_t)
    z_s, ctx_s = z_t[si], ctx_t[si]
    si_np = si.cpu().numpy()
    inv_si = np.zeros(T, dtype=int)
    inv_si[si_np] = np.arange(T)

    comp_net = _PerSampleNet(D_).to(device)
    sep_opt = _torch.optim.AdamW([w], lr=sep_lr, weight_decay=0.0)
    comp_opt = _torch.optim.AdamW(
        comp_net.parameters(), lr=comp_lr, weight_decay=1e-3
    )
    sep_sch = _torch.optim.lr_scheduler.CosineAnnealingLR(sep_opt, epochs)
    comp_sch = _torch.optim.lr_scheduler.CosineAnnealingLR(comp_opt, epochs)

    cached_mask = None
    cached_viol_idx = np.array([], int)
    cached_viol_w: _torch.Tensor | None = None

    comp_net.train()
    for ep in range(epochs):
        use_comp = ep >= compensation_delay

        # ----- Separation step (update W) -----
        s_raw = (w.unsqueeze(0) * z_s).sum(-1)
        s_raw_n = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-8)
        s_raw_n = s_raw_n.clamp(max=clamp_max)
        sep_nonlin = (
            s_raw_n.sign() * (nonlin_scale * s_raw_n).abs().pow(sep_exponent)
        )
        L_sep = -sep_nonlin.mean()

        sep_opt.zero_grad()
        L_sep.backward()
        _torch.nn.utils.clip_grad_norm_([w], 10.0)
        sep_opt.step()

        if not use_comp:
            sep_sch.step()
            comp_sch.step()
            continue

        # ----- Compensation step (update NN, W frozen) -----
        dw_raw, b = comp_net(ctx_s)
        dw_n = dw_raw.norm(dim=-1, keepdim=True) + 1e-8
        w_for_comp = w.detach().clone()
        with _torch.no_grad():
            max_a = max_corr * (float(w_for_comp.norm()) + 1e-8)
        dw = dw_raw * _torch.clamp(max_a / dw_n, max=1.0)
        w_eff = w_for_comp.unsqueeze(0) + dw
        s = (w_eff * z_s).sum(-1) + b.squeeze(-1)

        # Refresh spike mask + violations every 15 epochs
        if cached_mask is None or ep % 15 == 0:
            with _torch.no_grad():
                s_np = s.detach().cpu().numpy()
                cached_mask = gmm_cluster_mask(s_np)
                s_us = np.zeros(T)
                s_us[si_np] = s_np
                if distance_weighted:
                    viols, w_viol = _find_refractory_violations(
                        s_us, anchor_spikes, FS, return_weights=True
                    )
                else:
                    viols = _find_refractory_violations(s_us, anchor_spikes, FS)
                    w_viol = None
                if len(viols) > 0:
                    mask_ok = (viols >= 0) & (viols < T)
                    cached_viol_idx = inv_si[viols[mask_ok]]
                    cached_viol_w = (
                        _torch.tensor(
                            w_viol[mask_ok], dtype=_torch.float32, device=device
                        )
                        if distance_weighted and w_viol is not None
                        else None
                    )
                else:
                    cached_viol_idx = np.array([], int)
                    cached_viol_w = None

        spike_mask = _torch.tensor(cached_mask, dtype=_torch.bool, device=device)
        null_mask = ~spike_mask
        gmm_loss = _comp_loss(
            s, spike_mask, null_mask, kind=comp_loss,
            nonlin_scale=nonlin_scale, sep_exponent=sep_exponent,
            clamp_max=clamp_max,
        )

        if len(cached_viol_idx) > 0:
            abs_s = _torch.abs(s[cached_viol_idx])
            pow_s = abs_s ** refract_exponent
            if cached_viol_w is not None:
                suppress_loss = (cached_viol_w * pow_s).sum() / (
                    cached_viol_w.sum() + 1e-8
                )
            else:
                suppress_loss = pow_s.mean()
        else:
            suppress_loss = _torch.tensor(0.0, device=device)

        smooth = (
            _torch.mean((dw[1:] - dw[:-1]) ** 2)
            + _torch.mean((b[1:] - b[:-1]) ** 2)
        )

        L_comp = gmm_loss + lam_refract * suppress_loss + 0.5 * smooth
        comp_opt.zero_grad()
        L_comp.backward()
        _torch.nn.utils.clip_grad_norm_(comp_net.parameters(), 1.0)
        comp_opt.step()

        sep_sch.step()
        comp_sch.step()

    # Final forward pass
    comp_net.eval()
    with _torch.no_grad():
        dw_raw, b = comp_net(ctx_s)
        dw_n = dw_raw.norm(dim=-1, keepdim=True) + 1e-8
        max_a = max_corr * (w.detach().norm() + 1e-8)
        dw = dw_raw * _torch.clamp(max_a / dw_n, max=1.0)
        w_eff = w.detach().unsqueeze(0) + dw
        s_out = (w_eff * z_s).sum(-1) + b.squeeze(-1)
    s_full = np.zeros(T)
    s_full[si_np] = s_out.cpu().numpy()
    return s_full
