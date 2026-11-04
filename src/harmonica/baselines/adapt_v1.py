"""Mendez Guerra 2024 — Adapt baseline on the adapt_decomp **v1.0.0** engine.

Paper: *Adaptive EMG decomposition in dynamic conditions based on online
learning metrics with tunable hyperparameters*, J Neural Eng 21(4), 2024.
DOI `10.1088/1741-2552/ad5ebf`.

Sibling of :mod:`.adapt`, which wraps the *v0* vendored snapshot
(``vendored_adapt_decomp/``, upstream commit ``8ffeb66``).  This module wraps
the v1.0.0 release (``vendored_adapt_decomp_v1/``), a deliberately breaking
rewrite.

**Only the online adaptation engine changes.**  Calibration is this repo's own
ICA stack (``ica.run_ica_loop``) with the same Toeplitz + PCA + ZCA pipeline
every other baseline uses, exactly as :func:`.adapt.run_adapt` does, so a
difference between the ``adapt`` and ``adapt_v1`` rows measures the engine and
nothing else.  v1's own ``cbss/`` calibration stack is deliberately
*not* used: it would change the ICA, the contrast function and the duplicate
removal all at once, confounding the comparison.

The online engine itself runs **as vendored**: :func:`build_adapt_config`
starts from upstream's ``configs/adapt_configs/default_neuromotion.yaml`` and
only overrides the fields shared with the calibration (plus any explicitly
tuned ones, e.g. from ``scripts/tune_adapt_v1.py``).

Two v1 features let this wrapper drop hacks that :mod:`.adapt` needs:

* ``ext_mode="toeplitz"`` makes ``adapt_decomp``'s internal extension produce
  the same channel-major layout as :func:`..ica.utils.extension`, so the
  explicit delay-block permutation matrix ``P`` in :mod:`.adapt` is gone.
* ``Decomposition`` now takes ``pca_components``/``pca_mean`` alongside a
  *square* whitening matrix and applies ``(X_ext - pca_mean) @ components.T``
  itself.  :mod:`.adapt` instead folded PCA into a non-square whitening matrix,
  which silently dropped ``pca.mean_``; v1 rejects non-square whitening
  outright and handles the mean correctly.

The vendored v1 tree (``vendored_adapt_decomp_v1/``: upstream's ``src/``,
the NeuroMotion config, LICENSE, at commit ``a8f5373``) keeps upstream's
absolute ``adapt_decomp.*`` imports, so it cannot be imported as a relative
subpackage the way the v0 snapshot is.  Its ``src/`` is put on ``sys.path``
below instead.  The two versions occupy distinct top-level names
(``adapt_decomp`` vs ``harmonica.baselines.vendored_adapt_decomp``),
so importing this module does not perturb :mod:`.adapt`.
"""
from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

# --- Make the vendored v1 clone importable as the top-level `adapt_decomp` ---
_V1_ROOT = Path(__file__).parent / "vendored_adapt_decomp_v1"
_V1_SRC = _V1_ROOT / "src"
if str(_V1_SRC) not in sys.path:
    sys.path.insert(0, str(_V1_SRC))

# Upstream's NeuroMotion adaptation config: the "as vendored" defaults for the
# online engine (see build_adapt_config).
NEUROMOTION_YAML = _V1_ROOT / "configs" / "adapt_configs" / "default_neuromotion.yaml"

from adapt_decomp.adaptation.config import AdaptConfig            # noqa: E402
from adapt_decomp.adaptation.core import (                        # noqa: E402
    _SHARED_CBSS_ADAPT_FIELDS, AdaptDecomp,
)
from adapt_decomp.cbss.config import CBSSConfig                   # noqa: E402
from adapt_decomp.cbss.data_structure import CBSSResult           # noqa: E402
from adapt_decomp.preprocessing import preprocess_emg             # noqa: E402

# Reused verbatim from the v0 wrapper — v1 still detects spikes on
# |sources|**spike_det_exp with spike_det_exp=2.0, i.e. the same
# squared-amplitude domain _peak_centroids clusters in.
from .adapt import (                                              # noqa: E402
    _peak_centroids, _spike_matrix, merge_calib_online, online_start, online_switch,
    unpeeled_calib_sources,
)
from ..ica.algorithm import run_ica_loop                          # noqa: E402
from ..ica.utils import extension                                 # noqa: E402
from ..ica.whitening import whitening as _whiten                  # noqa: E402


# =============================================================================
# Calibration
# =============================================================================


def _cov_isi(spike_indices: np.ndarray, fs: float) -> float:
    """Coefficient of variation of the inter-spike intervals.

    Only carried on the :class:`CBSSResult` for provenance — nothing in the
    adaptation path reads it.

    Args:
        spike_indices: Sorted spike sample indices for one MU.
        fs: Sampling frequency, in Hz. Unused (the CoV is scale-free), kept
            so callers can pass it without special-casing.

    Returns:
        ``std(isi) / mean(isi)``, or ``nan`` with fewer than 3 spikes.
    """
    if len(spike_indices) < 3:
        return float("nan")
    isi = np.diff(np.sort(np.asarray(spike_indices, dtype=float)))
    mean_isi = isi.mean()
    if mean_isi <= 0:
        return float("nan")
    return float(isi.std() / mean_isi)


def calibrate_adapt_v1(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 2.0,
    ext_fact: int = 8,
    d_eff: int = 128,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    spike_det_exp: float = 2.0,
    verbose: bool = False,
) -> "tuple[CBSSResult, CBSSConfig] | tuple[None, CBSSConfig]":
    """Calibrate with this repo's ICA stack, packaged as a v1 ``CBSSResult``.

    Runs the identical Toeplitz + PCA + ZCA + FastICA block that
    :func:`.adapt.run_adapt` and every other baseline uses, then wraps the
    result in the dataclass ``adapt_decomp`` v1 expects, so that
    :meth:`AdaptDecomp.from_calibration` and
    ``adapt_decomp.adaptation.optimize``'s pooled searches both accept it.

    The returned :class:`CBSSConfig` describes what *this* calibration
    actually did, not what v1's own CBSS would have done. It exists because
    ``from_calibration`` treats it as ground truth for every field it shares
    with :class:`AdaptConfig` (``ext_fact``, ``ext_mode``, ``spike_det_exp``
    and the filter fields) and emits a ``UserWarning`` per call on any
    mismatch — so :func:`run_adapt_v1` builds its ``AdaptConfig`` with the
    same values.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first EMG, already
            bandpassed by the caller.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window length, in seconds (the first
            ``calib_s`` seconds of ``emg``).
        ext_fact: Time-delay extension factor.
        d_eff: PCA dimension cap after extension (capped at ``D_full - 1``).
        sil_th: SIL threshold for the calibration ICA.
        ica_seed: Calibration ICA RNG seed.
        spike_det_exp: Detection exponent: spikes are detected, and centroids
            and the IQR gate live, on ``|source|**spike_det_exp``. The
            calibration centroids are computed in that domain, and it is
            written into ``cbss_config`` so the engine uses the same one.
        verbose: Print the MU count and dimensions.

    Raises:
        ValueError: If ``calib_s`` exceeds the signal length.

    Returns:
        ``(calibration, cbss_config)``. ``calibration`` is ``None`` when ICA
        found no motor units; ``cbss_config`` is always returned.
    """
    emg = np.asarray(emg, dtype=np.float32)
    C, T = emg.shape

    T_calib = int(calib_s * FS)
    if T_calib >= T:
        raise ValueError(f"calib_s={calib_s}s exceeds signal length {T/FS:.2f}s")

    emg_calib = emg[:, :T_calib]                            # (C, T_calib)

    # 1. Toeplitz extension — same layout as the other baselines, and the
    #    layout AdaptConfig(ext_mode="toeplitz") reproduces online.
    z_ext_full = extension(emg_calib, ext_fact)             # (D_full, T_calib)
    D_full = z_ext_full.shape[0]

    # 2. PCA reduction
    d = min(d_eff, D_full - 1)
    # random_state pins sklearn's randomized SVD solver, which 'auto' selects
    # at these shapes; without it the reduction -- and so the whole calibration
    # -- is not reproducible even for a fixed ICA seed.
    pca = PCA(n_components=d, random_state=ica_seed).fit(z_ext_full.T)
    z_pca = pca.transform(z_ext_full.T).T                   # (d, T_calib)

    # 3. ZCA whitening in the reduced space
    z_w, Z_zca = _whiten(z_pca, method="ZCA")               # Z_zca: (d, d)

    # 4. ICA
    src_calib, spk_calib, sil_calib, filters = run_ica_loop(
        z_w, FS,
        sil_th=sil_th,
        n_iter=60,
        seed=ica_seed,
        refinement_loop=True,
        peel_off=True,
    )
    # filters: (d, n_mu) — column per MU in whitened d-dim space
    n_mu = int(src_calib.shape[0])
    # ICA's src_calib is peel-off residue; use what the online loop will see.
    # Feeds the centroids and CBSSResult.sources (IQR gate, contrast stats).
    if n_mu > 0:
        src_calib = unpeeled_calib_sources(filters, z_w)    # (n_mu, T_calib)

    cbss_config = CBSSConfig(
        fs=float(FS),
        ext_fact=ext_fact,
        # Matches ica.utils.extension's channel-major layout, so no
        # block -> Toeplitz permutation is needed anywhere.
        ext_mode="toeplitz",
        n_components=d,
        whitening_method="ZCA",
        spike_det_exp=spike_det_exp,
        spike_min_dist_ms=10.0,
        sil_th=sil_th,
        random_seed=ica_seed,
        # The caller (scripts/bench_adapt.py / tune_adapt_v1.py) has already
        # applied a zero-phase 10-1000 Hz Butterworth; these fields only
        # document that, since every call site passes preprocess=False.
        preprocess_emg=False,
        lowcut=10.0,
        highcut=1000.0,
        filter_order=2,
        powerline=False,
        device="cpu",
        verbose=verbose,
    )

    if n_mu == 0:
        return None, cbss_config

    if verbose:
        print(f"  [adapt_v1] calibration: {n_mu} MUs in {calib_s:.1f}s "
              f"(D={D_full}->d={d})")

    min_spike_dist = max(1, int(0.010 * FS))  # 10 ms refractory floor

    spikes_mat = np.stack(
        [_spike_matrix(np.asarray(spk_calib[j]), T_calib) for j in range(n_mu)],
        axis=1,
    ).astype(np.int32)                                      # (T_calib, n_mu)

    base_centrs = np.zeros(n_mu, dtype=np.float32)
    spike_centrs = np.zeros(n_mu, dtype=np.float32)
    for j in range(n_mu):
        base_centrs[j], spike_centrs[j] = _peak_centroids(
            src_calib[j], min_spike_dist, power=spike_det_exp,
        )

    calibration = CBSSResult(
        sources=np.ascontiguousarray(src_calib.T, dtype=np.float32),   # (T_calib, n_mu)
        spikes=spikes_mat,                                             # (T_calib, n_mu)
        spikes_dict={j: np.asarray(spk_calib[j]) for j in range(n_mu)},
        # CBSSResult stores (dim, n_mu); to_adapt_tensors() transposes it to
        # the (n_mu, dim) AdaptDecomp wants.
        sep_vectors=np.ascontiguousarray(filters, dtype=np.float32),   # (d, n_mu)
        whitening=np.ascontiguousarray(Z_zca, dtype=np.float32),       # (d, d)
        extension_mean=z_ext_full.mean(axis=1)[None, :].astype(np.float32),
        spikes_centr=spike_centrs,
        base_centr=base_centrs,
        sil=np.asarray(sil_calib, dtype=np.float32),
        cov_isi=np.asarray(
            [_cov_isi(spk_calib[j], FS) for j in range(n_mu)], dtype=np.float32
        ),
        ext_fact=ext_fact,
        # Passed through to Decomposition, which applies
        # (X_ext - pca_mean) @ pca_components.T per batch.
        pca_components=np.ascontiguousarray(pca.components_, dtype=np.float32),  # (d, D_full)
        pca_mean=np.ascontiguousarray(pca.mean_, dtype=np.float32),              # (D_full,)
        emg=np.ascontiguousarray(emg_calib.T, dtype=np.float32),       # (T_calib, C)
        timestamps=np.arange(T_calib, dtype=np.float64) / float(FS),
    )
    return calibration, cbss_config


# =============================================================================
# Public entry point
# =============================================================================


def run_adapt_v1(
    emg: np.ndarray,
    FS: float,
    calib_s: float = 2.0,
    ext_fact: int = 8,
    d_eff: int = 128,
    sil_th: float = 0.8,
    ica_seed: int = 1909,
    batch_ms: int | None = None,
    adapt_wh: bool | None = None,
    adapt_sv: bool | None = None,
    adapt_sd: bool | None = None,
    wh_learning_rate: float | None = None,
    sv_learning_rate: float | None = None,
    wh_sv_coupling: bool | None = None,
    lr_mode: str | None = None,
    wh_mode: str | None = None,
    contrast_scope: str | None = None,
    fifo_s: float | None = None,
    centroid_momentum: float | None = None,
    spike_det_exp: float = 2.0,
    device: str | None = None,
    verbose: bool = False,
    compute_loss: bool = False,
    return_loss: bool = False,
    bandpass: bool = True,
    online_start_s: float | None = None,
) -> "dict[int, np.ndarray] | tuple[dict[int, np.ndarray], float]":
    """Run Adapt (Mendez Guerra 2024, adapt_decomp v1.0.0) on a dynamic recording.

    The online engine runs **as vendored**: every adaptation setting comes from
    upstream's ``configs/adapt_configs/default_neuromotion.yaml`` (see
    :func:`build_adapt_config`) unless overridden here. Every adaptation
    argument below defaults to ``None``, meaning "take the NeuroMotion value".
    The calibration (the shared Toeplitz + PCA + ZCA + FastICA pipeline, see
    :func:`calibrate_adapt_v1`) and its shared fields (``ext_fact``,
    ``ext_mode``, filters, ``spike_det_exp``) stay this repo's.

    As in :func:`.adapt.run_adapt` (and Chen / Yeung), the online loop starts
    on the batch right after calibration and the calibration ICA's own
    spikes are reported on ``[0, T_calib)``.

    Args:
        emg: ``(n_channels, n_samples)`` channels-first raw EMG.
        FS: Sampling frequency, in Hz.
        calib_s: Calibration window length, in seconds. Paper uses ``1-2 s``.
        ext_fact: Time-delay extension factor (the benchmark uses 8 uniformly
            across baselines; NeuroMotion's 10 is a calibration field, so
            ``cbss_config`` wins).
        d_eff: PCA dimension cap after extension (must be < C x ext_fact).
        sil_th: SIL threshold for the calibration ICA.
        ica_seed: Calibration ICA RNG seed.
        batch_ms, adapt_wh, adapt_sv, adapt_sd, wh_learning_rate,
        sv_learning_rate, wh_sv_coupling, lr_mode, wh_mode, contrast_scope,
        centroid_momentum: ``AdaptConfig`` overrides; ``None`` keeps the
            NeuroMotion value (batch 100 ms, all three adaptations on,
            learning rates 7e-3 / 3e-3, no coupling, ``"fixed"``,
            ``"kl_to_identity"``, ``"spike_based"``, momentum 0.95).
        fifo_s: Whitening-covariance FIFO length in seconds (converted to
            ``fifo_length`` samples). ``None`` keeps NeuroMotion's
            ``fifo_length: null`` = ``2 * d`` samples. A value above
            ``calib_s - batch_ms/1000`` leaves < 2 calibration windows for the
            whitening-loss normaliser (warned; the ``lr_mode="fixed"`` update
            itself is unaffected).
        spike_det_exp: Detection exponent (a calibration field): peaks,
            centroids, the midpoint threshold and the IQR gate all use
            ``|source|**spike_det_exp``. 2.0 = library default.
        device: ``'cuda'`` / ``'cpu'`` / ``None`` (auto).
        verbose: Print calibration progress.
        compute_loss: Compute the online adaptation loss (wh + sv). Adds a
            small overhead per batch. Required for ``return_loss=True``.
        return_loss: If True, return ``(spike_dict, total_loss)`` instead of
            just the spike dict. ``compute_loss`` is enabled automatically.
            v1's ``total_loss`` is a single scalar:
            ``median(wh_loss) + median(sv_loss.nansum(dim=1))``, or the
            sentinel ``1e10`` when the whitening diverged.
        bandpass: Apply an internal causal 20-500 Hz bandpass before
            processing. Set to ``False`` when the caller has already filtered
            (e.g. 10-1000 Hz in the benchmark loop) to avoid double-filtering.
        online_start_s: Where the online loop starts, in seconds (capped at
            ``calib_s``). ``None`` (default) starts right after calibration;
            ``0.0`` replays the whole recording through the online engine.
            See :func:`.adapt.run_adapt`.

    Raises:
        ValueError: If ``calib_s`` exceeds the signal length.

    Returns:
        ``{mu_id: sorted spike sample indices}`` when ``return_loss=False``,
        ``({mu_id: ...}, float)`` when ``return_loss=True``. MUs with fewer
        than 5 spikes are dropped, as in :func:`.adapt.run_adapt`.
    """
    emg = np.asarray(emg, dtype=np.float32)

    # ---- Bandpass (skip if the caller already filtered the signal) -------
    # v1 has no standalone bandpass_filter; preprocess_emg is the shared path
    # and filters along axis=0, so it needs samples-first input.
    if bandpass:
        emg_cf = preprocess_emg(
            emg.T, fs=float(FS),
            highpass=20.0, lowpass=500.0, filter_order=4,
            notch_filter=False,
        ).T.astype(np.float32)          # back to (C, T)
    else:
        emg_cf = emg                    # (C, T) — already filtered by caller

    calibration, cbss_config = calibrate_adapt_v1(
        emg_cf, FS,
        calib_s=calib_s, ext_fact=ext_fact, d_eff=d_eff,
        sil_th=sil_th, ica_seed=ica_seed, spike_det_exp=spike_det_exp,
        verbose=verbose,
    )
    if calibration is None:
        return ({}, float("inf")) if return_loss else {}

    n_mu = calibration.sources.shape[1]

    overrides = {k: v for k, v in dict(
        batch_ms=batch_ms, adapt_wh=adapt_wh, adapt_sv=adapt_sv, adapt_sd=adapt_sd,
        wh_learning_rate=wh_learning_rate, sv_learning_rate=sv_learning_rate,
        wh_sv_coupling=wh_sv_coupling, lr_mode=lr_mode, wh_mode=wh_mode,
        contrast_scope=contrast_scope, centroid_momentum=centroid_momentum,
    ).items() if v is not None}
    adapt_config = build_adapt_config(
        cbss_config,
        fifo_s=fifo_s,
        compute_loss=compute_loss or return_loss,   # return_loss implies compute_loss
        device=device,
        **overrides,
    )
    if fifo_s is not None and fifo_s > calib_s - adapt_config.batch_ms / 1000:
        warnings.warn(
            f"fifo_s={fifo_s}s leaves fewer than two {adapt_config.batch_ms} ms-strided "
            f"calibration windows in calib_s={calib_s}s: the whitening loss "
            "normaliser falls back to 1e-7, so wh_loss/total_loss (and "
            "lr_mode='rel_error') are meaningless.",
            RuntimeWarning,
            stacklevel=2,
        )

    model = AdaptDecomp.from_calibration(
        calibration=calibration,
        cbss_config=cbss_config,
        adapt_config=adapt_config,
    )
    # By default the online loop starts on the batch right after calibration:
    # the slice begins ext_fact samples early so the first scored sample
    # (batch 0 drops ext_fact rows) is t_on = T_calib -- same convention as
    # adapt.run_adapt.
    T_calib = calibration.emg.shape[0]
    t_on = online_switch(T_calib, FS, online_start_s)
    s_on = online_start(t_on, ext_fact)
    outputs = model.process_data(
        torch.from_numpy(np.ascontiguousarray(emg_cf[:, s_on:].T)),   # (T - s_on, C)
        preprocess=False,
        processing_mode="offline",
    )

    # (T - s_on, n_mu) int32, relative to s_on.  Calibration ICA spikes cover
    # [0, t_on).
    spikes_mat = outputs.spikes.cpu().numpy()
    min_gap = max(1, int(0.010 * FS))
    result: dict[int, np.ndarray] = {}
    for j in range(n_mu):
        idx = merge_calib_online(
            calibration.spikes_dict[j],
            np.flatnonzero(spikes_mat[:, j]) + s_on,
            t_on, min_gap,
        )
        if len(idx) >= 5:
            result[j] = idx

    if return_loss:
        return result, float(outputs.total_loss)
    return result


def build_adapt_config(
    cbss_config: CBSSConfig,
    *,
    fifo_s: float | None = None,
    compute_loss: bool = True,
    device: str | None = None,
    **overrides,
) -> AdaptConfig:
    """Upstream's NeuroMotion ``AdaptConfig``, made consistent with ``cbss_config``.

    Starts from ``AdaptConfig.from_yaml(NEUROMOTION_YAML)`` -- the vendored
    ``configs/adapt_configs/default_neuromotion.yaml`` -- so every adaptation
    setting (learning rates, ``lr_mode``, ``wh_mode``, ``wh_sv_coupling``,
    ``contrast_scope``, ``centroid_momentum``, ``fifo_length``, shrinkage,
    clipping, EMA constants, ...) is exactly the packaged one unless
    overridden. Then:

    * the 15 fields ``AdaptDecomp.from_calibration`` reconciles with the
      calibration (``_SHARED_CBSS_ADAPT_FIELDS``: ``ext_fact``, ``ext_mode``,
      ``spike_det_exp``, ``fs``, filter/notch fields, channel masks) are taken
      from ``cbss_config``, so that reconciliation is a no-op and does not
      warn on every call;
    * ``overrides`` (any other ``AdaptConfig`` field, e.g. tuned learning
      rates or ``centroid_momentum``) are applied last.

    ``dataclasses.replace`` re-runs ``__post_init__``, so the derived
    ``batch_size`` / ``spike_min_dist`` follow ``batch_ms`` / ``fs``. Shared by
    :func:`run_adapt_v1` and ``scripts/tune_adapt_v1.py``, so tuned values are
    applied on the test set under exactly the config they were tuned under.

    Args:
        cbss_config: The config returned by :func:`calibrate_adapt_v1`.
        fifo_s: Whitening-covariance FIFO length, in seconds; converted to
            ``fifo_length`` samples with ``cbss_config.fs``. ``None`` keeps the
            YAML's ``fifo_length`` (``null`` = ``2 * d``).
        compute_loss: Compute the online adaptation loss each batch.
        device: ``'cuda'`` / ``'cpu'`` / ``None`` (auto).
        **overrides: ``AdaptConfig`` field overrides (not shared fields).

    Raises:
        ValueError: If an override names a shared calibration field or an
            unknown ``AdaptConfig`` field.

    Returns:
        AdaptConfig: Ready for ``AdaptDecomp.from_calibration``.
    """
    valid = {f.name for f in dataclasses.fields(AdaptConfig) if f.init}
    bad = sorted(k for k in overrides if k not in valid or k in _SHARED_CBSS_ADAPT_FIELDS)
    if bad:
        raise ValueError(f"not overridable here (unknown or shared with cbss_config): {bad}")
    shared = {f: getattr(cbss_config, f) for f in _SHARED_CBSS_ADAPT_FIELDS}
    shared["fs"] = int(cbss_config.fs)
    if fifo_s is not None:
        overrides["fifo_length"] = int(round(fifo_s * cbss_config.fs))
    base = AdaptConfig.from_yaml(NEUROMOTION_YAML)
    return dataclasses.replace(
        base, **shared, **overrides,
        compute_loss=compute_loss, device=device, save_params=False,
    )
