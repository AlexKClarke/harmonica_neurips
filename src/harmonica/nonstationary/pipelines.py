"""End-to-end pipelines — preprocess, run_ica, refine_*, run_all_methods.

Thin wrappers around the primitives in
:mod:`harmonica.ica` and :mod:`harmonica.nonstationary`.

All refine_* functions return a refined ``source`` signal (not spikes).
Spike extraction lives in ``detect.detect_best`` so the caller can choose
a detector that fits their use-case.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ..evaluate import evaluate_decomposition
from ..ica.algorithm import run_ica_loop
from ..ica.utils import extension
from ..ica.whitening import whitening
from .detect import detect_best
from .harmonica import iternn_refine, persample_refine
from .refractory import refine_harmonica_style, refine_iterative_refractory


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
    # random_state pinned: at these dimensions sklearn auto-selects the
    # randomized SVD solver, which is non-deterministic without a seed (two
    # identical calls differed by up to 12.2). This propagated into every
    # filter, source and spike train.
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
            STA peel-off between sources. Default ``False`` keeps the
            historical behaviour (this wrapper has always disabled it).
            Set to ``True`` for the "default peel-off in ICA" ablation.

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
# Method wrappers
# =============================================================================


def refine_timenn(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    epochs: int = 200,
    mc: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Single-pass per-sample NN refinement (``pure_gmm`` loss)."""
    return persample_refine(
        z_w, w_ica, t_norm,
        loss_type="pure_gmm", epochs=epochs, max_corr=mc, seed=seed,
    )


def refine_iternn(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    mc_schedule: list[float] | None = None,
    ep_schedule: list[int] | None = None,
    select: str = "last",
    seed: int = 42,
) -> np.ndarray:
    """Multi-pass NN refinement with STA re-estimation."""
    if mc_schedule is None:
        mc_schedule = [0.3, 0.4, 0.5]
    if ep_schedule is None:
        ep_schedule = [150, 150, 200]
    _, _, source, _ = iternn_refine(
        z_w, w_ica, t_norm,
        mc_schedule=mc_schedule, ep_schedule=ep_schedule,
        loss_type="pure_gmm", fsamp=FS, seed=seed, select=select,
    )
    return source


# =============================================================================
# NN-guided peel-off + peel-then-refine pipeline
# =============================================================================


def nn_guided_peeloff(
    z_w: np.ndarray,
    spk: dict,
    filters: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    T: int,
    target_eid: int,
    n_bins: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Per-bin STA peel-off of every non-target source.

    For each non-target source:

    1. Run short ``iternn_refine`` (50 epochs × 3 passes) to capture
       time-varying filter behaviour.
    2. Detect spikes on the NN source.
    3. Bin spikes by time, compute per-bin STA, subtract at each spike.

    Accounts for template rotation across time — unlike a single-STA peel.

    Args:
        z_w: ``(D, T)`` whitened signal.
        spk: ``{est_id: spike_indices}`` from ICA.
        filters: ``(D, n_sources)`` ICA filter matrix.
        t_norm: ``(T,)`` normalised time context.
        FS: Sampling frequency, in Hz.
        T: Signal length in samples.
        target_eid: ``est_id`` to preserve (not peeled).
        n_bins: Number of time bins per source.
        seed: RNG seed for the inner ``iternn_refine``.

    Returns:
        ``(D, T)`` whitened signal with non-target sources peeled.
    """
    from ..ica.detect import est_spike_times

    z_clean = z_w.copy()
    for eid in range(filters.shape[1]):
        if eid == target_eid or eid not in spk:
            continue
        sp = spk[eid].astype(int)
        sp = sp[(sp >= 0) & (sp < T)]
        if len(sp) < 10:
            continue
        w = filters[:, eid] / np.linalg.norm(filters[:, eid])
        try:
            _, _, s_nn, _ = iternn_refine(
                z_clean, w, t_norm,
                mc_schedule=[0.3, 0.4, 0.5], ep_schedule=[50, 50, 50],
                fsamp=FS, seed=seed, select="last",
            )
        except Exception:
            continue
        sp_nn, _ = est_spike_times(s_nn, FS)
        sp_nn = sp_nn[(sp_nn >= 0) & (sp_nn < T)].astype(int)
        if len(sp_nn) < 5:
            continue
        bin_edges = np.linspace(t_norm.min(), t_norm.max(), n_bins + 1)
        for b in range(n_bins):
            mask = (t_norm[sp_nn] >= bin_edges[b]) & (
                t_norm[sp_nn] < bin_edges[b + 1]
            )
            sp_bin = sp_nn[mask]
            if len(sp_bin) < 3:
                continue
            sta_bin = z_clean[:, sp_bin].mean(axis=1)
            for t in sp_bin:
                z_clean[:, t] -= sta_bin
    return z_clean


def refine_peeloff_timenn(
    z_w: np.ndarray,
    w_ica: np.ndarray,
    t_norm: np.ndarray,
    FS: float,
    T: int,
    spk: dict,
    filters: np.ndarray,
    target_eid: int,
    epochs: int = 200,
    mc: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Peel non-target MUs via :func:`nn_guided_peeloff`, then :func:`refine_timenn`."""
    z_peeled = nn_guided_peeloff(
        z_w, spk, filters, t_norm, FS, T,
        target_eid=target_eid, seed=seed,
    )
    return persample_refine(
        z_peeled, w_ica, t_norm,
        loss_type="pure_gmm", epochs=epochs, max_corr=mc, seed=seed,
    )


# =============================================================================
# End-to-end benchmark runner
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


def run_all_methods(
    emg: np.ndarray,
    FS: float,
    gt: dict,
    methods: list[str] | None = None,
    R: int = 4,
    d_eff: int = 128,
    ica_seed: int = 1909,
    nn_seed: int = 42,
) -> dict:
    """Run multiple refinement methods on a recording, evaluate against GT.

    For each GT MU matched by ICA, reports per-method F1 plus the
    :func:`proj_ratio_guardrail` fallback numbers (``*_guard_f1``,
    ``*_proj_ratio``, ``*_spike_ratio``, ``*_guard_trig``). The guardrail
    lives in ``models/guardrails_zoo`` — pulled in lazily so that the
    core library does not depend on the zoo.

    Args:
        emg: ``(n_channels, n_samples)`` raw EMG.
        FS: Sampling frequency, in Hz.
        gt: ``{mu_id: spike_indices}`` ground truth.
        methods: Subset of ``['timenn', 'iternn', 'iterative_refractory',
            'harmonica_style', 'peeloff_timenn']``. Defaults to the first
            four (the main refinement path).
        R: Extension factor.
        d_eff: PCA components.
        ica_seed: ICA RNG seed.
        nn_seed: NN RNG seed.

    Returns:
        ``{'detected_mus': [...], gid: per-method-metrics-dict, ...}``.
    """
    if methods is None:
        methods = ["timenn", "iternn", "iterative_refractory", "harmonica_style"]

    # lazy import — guardrail is in the zoo
    try:
        from ..models.guardrails_zoo import proj_ratio_guardrail
    except ImportError:
        proj_ratio_guardrail = None  # type: ignore[assignment]

    T = emg.shape[1]
    z_w, _ = preprocess(emg, R=R, d_eff=d_eff)
    spk, _sils, filters, t_norm = run_ica(z_w, FS, len(gt), seed=ica_seed)
    ev = evaluate_decomposition(spk, gt, FS)

    best_per_mu: dict[int, dict[str, Any]] = {}
    for ps in ev["per_source"]:
        gid = ps["gt_id"]
        if gid >= 0 and (gid not in best_per_mu or ps["f1"] > best_per_mu[gid]["f1"]):
            best_per_mu[gid] = ps

    results: dict[str, Any] = {"detected_mus": list(best_per_mu.keys())}

    def _guard(ica_sp: np.ndarray, sp: np.ndarray, w_ica: np.ndarray):
        """Call guardrail if available, else return (sp, no-op info)."""
        if proj_ratio_guardrail is None:
            return sp, {
                "proj_ratio": 1.0, "spike_ratio": 1.0, "triggered": False,
            }
        return proj_ratio_guardrail(ica_sp, sp, w_ica, z_w)

    def _store(r: dict, prefix: str, sp, sp_g, info, gt_sp):
        p, rec, f1 = _prf(sp, gt_sp, FS)
        r[f"{prefix}_p"], r[f"{prefix}_r"], r[f"{prefix}_f1"] = p, rec, f1
        p, rec, f1 = _prf(sp_g, gt_sp, FS)
        r[f"{prefix}_guard_p"] = p
        r[f"{prefix}_guard_r"] = rec
        r[f"{prefix}_guard_f1"] = f1
        r[f"{prefix}_proj_ratio"] = info["proj_ratio"]
        r[f"{prefix}_spike_ratio"] = info["spike_ratio"]
        r[f"{prefix}_guard_trig"] = info["triggered"]

    for gid, ps in best_per_mu.items():
        eid = ps["est_id"]
        w_ica = filters[:, eid] / np.linalg.norm(filters[:, eid])
        ica_sp = spk[eid].astype(int)
        r: dict[str, Any] = {
            "ica_p": ps["precision"], "ica_r": ps["recall"], "ica_f1": ps["f1"],
        }

        if "timenn" in methods:
            s = refine_timenn(z_w, w_ica, t_norm, FS, seed=nn_seed)
            sp, _ = detect_best(s, FS, T)
            sp_g, info = _guard(ica_sp, sp, w_ica)
            _store(r, "timenn", sp, sp_g, info, gt[gid])

        if "iternn" in methods:
            s = refine_iternn(z_w, w_ica, t_norm, FS, seed=nn_seed)
            sp, _ = detect_best(s, FS, T)
            sp_g, info = _guard(ica_sp, sp, w_ica)
            _store(r, "iternn", sp, sp_g, info, gt[gid])

        if "iterative_refractory" in methods:
            s = refine_iterative_refractory(
                z_w, w_ica, t_norm, FS, ica_sp, seed=nn_seed,
            )
            sp, _ = detect_best(s, FS, T)
            sp_g, info = _guard(ica_sp, sp, w_ica)
            _store(r, "refractory", sp, sp_g, info, gt[gid])

        if "harmonica_style" in methods:
            s = refine_harmonica_style(
                z_w, w_ica, t_norm, FS, ica_sp, seed=nn_seed,
            )
            sp, _ = detect_best(s, FS, T)
            sp_g, info = _guard(ica_sp, sp, w_ica)
            _store(r, "harm_style", sp, sp_g, info, gt[gid])

        if "peeloff_timenn" in methods:
            s = refine_peeloff_timenn(
                z_w, w_ica, t_norm, FS, T, spk, filters, eid, seed=nn_seed,
            )
            sp, _ = detect_best(s, FS, T)
            sp_g, info = _guard(ica_sp, sp, w_ica)
            _store(r, "peeloff_timenn", sp, sp_g, info, gt[gid])

        results[gid] = r

    return results
