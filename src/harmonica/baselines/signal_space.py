"""HarmonICA — the method of the paper (``--method signal_space``).

Signal-space compensation with neural peel-off, warm-started from FastICA.
Each motor unit gets a separation vector ``w_j`` and a low-rank
time-varying transform ``A_j(t) = I + U_j(t) V_j^T`` (rank 3 by default,
see :class:`...engine.configs.SurfaceConfig`) applied to the whitened
signal. Training alternates an independence step on ``w_j`` (uncompensated
source) with a compensation step on ``A_j`` (``w_j`` detached), the
latter switched on after ``sep_only_warmup`` epochs. After each unit the
compensated spike-triggered average is pushed back through ``A_j(t)⁻¹``
and subtracted from the signal (neural peel-off).

The per-unit variant without peel-off is
:func:`...models.harmonica_signal_space.refine_signal_space_comp_warmstart`.

Pipeline:

1. Preprocess raw EMG (Toeplitz extension + PCA + ZCA whitening). No
   bandpass is applied here; recordings are used as provided.
2. Run :func:`...ica.algorithm.run_ica_loop` on the whitened signal to
   discover ``K`` filters (``peel_off=False`` — Gram-Schmidt deflation
   gives independent filters in the whitened frame; we use neural
   peel-off post-refinement instead).
3. Build a single persistent :class:`...engine.state.State` on
   the whitened signal. ``state.signal`` accumulates peels across MUs.
4. For each filter ``k`` in ICA-discovery order:

    a. Re-init ``state.compensation_weights`` (fresh NN per MU).
    b. Override ``state.separation_weights`` with our ``w_ica[k]``.
    c. Re-init optimisers (they reference parameter handles).
    d. Run ``epochs`` of alternating ``joint_step`` (sep + comp).
    e. Detect spikes from compensated source.
    f. Call ``model.peel_off(timestamps)`` — computes STA in the
       compensated frame, inverse-transforms via ``A(t)⁻¹`` per spike,
       subtracts from ``state.signal`` in place.

5. Return ``{k: spike_indices}``.

Output contract matches the other baselines so it slots into
:mod:`scripts.bench_baselines` directly.
"""
from __future__ import annotations

import numpy as np


def run_signal_space(
    emg: np.ndarray,
    FS: float,
    angle: np.ndarray | None = None,
    *,
    epochs: int = 200,
    sep_only_warmup: int = 50,
    R: int = 4,
    d_eff: int = 128,
    n_iter_ica: int | None = None,
    sil_th_ica: float = 0.8,
    ica_seed: int = 1909,
    seed: int | None = 42,
    warmstart_w: bool = True,
    freeze_w: bool = False,
    peel_kind: str = "neural",
    use_angle: bool = False,
    couple_grads: bool = False,
    config_overrides: dict | None = None,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Multi-MU signal-space refinement with neural peel-off.

    Args:
        emg: ``(n_channels, n_samples)`` raw channels-first EMG.
        FS: Sampling frequency, in Hz.
        angle: Optional ``(n_samples,)`` joint-angle context. Only used
            when ``use_angle=True``. Ignored otherwise.
        use_angle: If True, pass ``angle`` as auxiliary to the comp net
            ("HarmonICA (θ)"). Default False uses the engine's built-in
            sinusoidal positional encoding of time ("HarmonICA (PE)").
        epochs: Number of alternating ``joint_step`` calls per MU.
        sep_only_warmup: Number of initial epochs in which only the
            separation vector is updated (compensation NN delayed).
        R: Toeplitz extension factor for preprocessing. The paper uses
            ``R=8``; the default here is 4.
        d_eff: PCA target dim post-extension.
        n_iter_ica: Max sources to extract in ICA. ``None`` →
            ``min(n_channels * 3, 108)`` (matches the
            :func:`...nonstationary.pipelines.run_ica` heuristic).
        sil_th_ica: SIL threshold for ICA acceptance.
        ica_seed: ICA RNG seed.
        seed: Optional ``torch.manual_seed`` for the comp-net inits.
        warmstart_w: Initialise ``w_j`` from the ICA filter (default).
            False draws a random unit vector (cold-start ablation).
        freeze_w: Keep ``w_j`` fixed at its initialisation and train only
            the compensator (ablation; the paper method trains both).
        peel_kind: ``'neural'`` (default), ``'classical'`` (STA peel-off
            in the whitened frame) or ``'none'``.
        couple_grads: Compute the independence loss on the compensated
            source, so it also reaches the compensator (ablation).
        config_overrides: Optional dict of :class:`SurfaceConfig`
            override kwargs (e.g. ``{"compensation_rank": 5}``).
        verbose: Per-MU progress print.

    Returns:
        ``{mu_id: spike_indices}`` — keyed by ICA filter index, in the
        sample-index space of the original recording.
    """
    import torch

    from ..ica.algorithm import run_ica_loop
    from ..ica.utils import extension
    from ..ica.whitening import whitening
    from ..models.engine.configs import SurfaceConfig
    from ..models.engine.decomposition import DecompositionModule
    from ..models.engine.state import State
    from sklearn.decomposition import PCA

    if seed is not None:
        torch.manual_seed(seed)

    C, T = emg.shape
    if n_iter_ica is None:
        n_iter_ica = min(C * 3, 108)

    # ---------- Preprocess ------------------------------------------------
    z_ext = extension(emg, R)
    d = min(d_eff, z_ext.shape[0] - 1)
    # random_state pinned so reruns are repeatable (the paper's runs left it unset).
    z_pca = PCA(n_components=d, random_state=0).fit_transform(z_ext.T).T
    z_w, _ = whitening(z_pca, method="ZCA")  # (D, T)
    D_, T_w = z_w.shape

    # ---------- ICA discovery (no STA peel-off; Gram-Schmidt deflation) ---
    _, _, _, filters = run_ica_loop(
        z_w, FS, sil_th=sil_th_ica, n_iter=n_iter_ica,
        seed=ica_seed, refinement_loop=True, peel_off=False,
    )
    K = filters.shape[1]
    if K == 0:
        return {}

    # ---------- Build persistent engine State -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SurfaceConfig(sampling_frequency=float(FS))
    # Disable the engine's gates — we keep peel-off but bypass SIL gating so we
    # always emit spikes for every filter we warm-start.
    config.acceptance_silhouette = 0.0
    config.min_sil_for_compensation = 0.0
    config.min_peaks_in_source = 1
    config.verbose_mode = False
    config.log_every_n_steps = 0
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # The engine's State expects (T, C)
    signal_t = torch.tensor(z_w.T, dtype=torch.float32, device=device)
    # Only use angle as auxiliary when explicitly enabled. Default
    # (use_angle=False) leaves auxiliary=None so the State falls back
    # to its sinusoidal positional encoding of time.
    aux = None
    if use_angle and angle is not None:
        aux = torch.tensor(np.asarray(angle, dtype=np.float32).reshape(-1, 1),
                           device=device)
    state = State(signal_t, config, auxiliary=aux)
    state.init_exponent()

    model = DecompositionModule()
    model.config = config
    model.state = state
    model._comp_step_count = 0
    model._cached_spike_ts = None

    # Coupled-gradient ablation: replace joint_step with a variant
    # that allows the ICA loss gradient to flow through the comp NN
    # (compensated=True for the L_sep source). The default keeps the two
    # losses decoupled (compensated=False); coupling lets L_sep drive
    # the comp NN, which is expected to game the ICA contrast.
    if couple_grads:
        import types

        def _coupled_joint_step(self, use_compensation: bool):
            self.train()
            self.state.separation_optimiser.zero_grad()
            self.state.compensation_optimiser.zero_grad()

            # Differs from the default: compensated=use_compensation (default False).
            # When use_compensation is True (post-warmup), the comp NN is
            # in the L_sep computation graph and receives ICA gradients.
            source = self.calculate_source(
                self.state.signal, compensated=use_compensation
            )

            sep_source = (source - source.mean()) / source.std()
            sep_source = sep_source.clamp(max=self.config.clamp_max)
            sep_nonlin = sep_source.sign() * (
                self.config.nonlin_scale * sep_source
            ).abs().pow(self.state.exponent_tensor)
            L_sep = -sep_nonlin.mean()
            L_sep.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.state.separation_weights], 10.0
            )
            self.state.separation_optimiser.step()

            isi = self.state.signal.shape[0]
            if use_compensation:
                self._comp_step_count = getattr(self, "_comp_step_count", 0) + 1
                resample = (
                    self._comp_step_count
                    % self.config.compensation_resample_every == 1
                    or self.config.compensation_resample_every == 1
                )
                if resample:
                    with torch.no_grad():
                        comp_source = self.calculate_source(
                            self.state.signal, compensated=True
                        )
                        comp_source = (
                            comp_source - comp_source.mean()
                        ) / comp_source.std()
                else:
                    comp_source = None

                samples = self.sample_spikes_and_nulls(
                    source=comp_source, reuse_cached=not resample
                )
                if samples is not None:
                    spike_vectors, spike_aux, null_vectors, null_aux, isi = samples
                    with torch.no_grad():
                        w = self.state.separation_weights.detach().clone()
                    comp_spike_vectors = self.state.compensation_weights.forward(
                        spike_vectors, spike_aux, use_cache=False
                    )
                    comp_null_vectors = self.state.compensation_weights.forward(
                        null_vectors, null_aux, use_cache=False
                    )
                    spike_source = torch.matmul(comp_spike_vectors, w)
                    null_source = torch.matmul(comp_null_vectors, w)
                    spike_std = spike_source.std()
                    null_std = null_source.std()
                    uncomp_null_var = torch.matmul(null_vectors, w).var()
                    L_comp = (
                        spike_std.square()
                        + torch.maximum(null_std.square(), uncomp_null_var)
                    ).sqrt()
                    L_comp.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.state.compensation_weights.parameters(),
                        max_norm=0.1,
                    )
                    self.state.compensation_optimiser.step()
                    self.state.update_compensation_cache()

            return isi

        model.joint_step = types.MethodType(_coupled_joint_step, model)

    spikes_dict: dict[int, np.ndarray] = {}

    # ---------- Per-MU loop with neural peel-off --------------------------
    for k in range(K):
        w_k = filters[:, k] / (np.linalg.norm(filters[:, k]) + 1e-12)

        # Per-MU re-init: separation warm-started from ICA, comp net fresh.
        # Cold-start ablation: replace warm-start with a random unit vector.
        if warmstart_w:
            w_init_t = torch.tensor(w_k, dtype=torch.float32, device=device)
        else:
            w_init_t = torch.randn(D_, device=device)
            w_init_t = w_init_t / (w_init_t.norm() + 1e-12)
        state.separation_weights = torch.nn.Parameter(w_init_t)
        state.init_compensation_weights()
        state.init_optimisers()
        # freeze_w=True zeroes the separation-optimizer LR so each
        # joint_step's sep update is a no-op (W stays at the init).
        # We do this post-init_optimisers rather than via requires_grad=False
        # so that L_sep.backward() doesn't error out on a frozen leaf.
        if freeze_w:
            for g in state.separation_optimiser.param_groups:
                g["lr"] = 0.0
        model._comp_step_count = 0
        model._cached_spike_ts = None

        # Train (alternating sep + comp).
        broke = False
        model.train()
        for ep in range(epochs):
            use_comp = ep >= sep_only_warmup
            try:
                model.joint_step(use_compensation=use_comp)
            except Exception as e:
                if verbose:
                    print(f"  [signal_space] MU {k} joint_step error at ep {ep}: "
                          f"{type(e).__name__}: {e}")
                broke = True
                break
        if broke:
            continue

        # Detect spikes from compensated source.
        try:
            with torch.no_grad():
                model.eval()
                timestamps = model.calculate_timestamps(calculate_silhouette=False)
                # calculate_timestamps returns a tensor or (ts, sil) tuple
                if isinstance(timestamps, tuple):
                    timestamps = timestamps[0]
        except Exception as e:
            if verbose:
                print(f"  [signal_space] MU {k} timestamp error: "
                      f"{type(e).__name__}: {e}")
            continue

        if len(timestamps) == 0:
            continue

        spikes_dict[k] = timestamps.detach().cpu().numpy().astype(int)

        # peel_kind chooses how to subtract the MU's contribution
        # from the persistent state.signal before the next MU trains.
        try:
            with torch.no_grad():
                if peel_kind == "neural":
                    # Compensated-frame STA, pushed back through A(t)^-1.
                    model.peel_off(timestamps)
                elif peel_kind == "classical":
                    # Classical FFT/STA peel on the (T,D) signal.
                    from ..ica.algorithm import peel_off as _peel_classical
                    z_w_np = state.signal.detach().cpu().numpy().T  # (D, T)
                    ts_np = timestamps.detach().cpu().numpy().astype(int)
                    z_w_new, _, _ = _peel_classical(z_w_np, ts_np, fsamp=FS)
                    state.signal = torch.tensor(
                        z_w_new.T, dtype=state.signal.dtype, device=device
                    )
                elif peel_kind == "none":
                    pass
                else:
                    raise ValueError(f"peel_kind must be one of "
                                     f"'neural'|'classical'|'none', got {peel_kind!r}")
        except Exception as e:
            if verbose:
                print(f"  [signal_space] MU {k} peel ({peel_kind}) error: "
                      f"{type(e).__name__}: {e}")
            # Don't drop the spikes; just stop the peel-off chain.
            break

        if verbose:
            print(f"  [signal_space] MU {k}: {len(timestamps)} spikes "
                  f"({peel_kind}-peeled).")

    return spikes_dict
