"""Smoke tests: the package imports, the detector is sign-preserving, and
HarmonICA and the v1.0.0 Adapt baseline run end to end on a small synthetic
convolutive mixture.

``test_benchmark_recording`` additionally runs the paper configuration on one
benchmark recording when ``HARMONICA_DATA`` points at ``benchmark_v2/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import harmonica  # import before torch (see README, "Troubleshooting")
from harmonica import est_spike_times, evaluate_decomposition, run_harmonica


def _best_f1(spikes: dict, gt: dict, fs: float) -> dict[int, float]:
    best: dict[int, float] = {}
    for ps in evaluate_decomposition(spikes, gt, fs)["per_source"]:
        g = ps["gt_id"]
        if g >= 0:
            best[g] = max(best.get(g, 0.0), ps["f1"])
    return best


def _synthetic_mixture(seed: int = 0, n_ch: int = 16, fs: int = 2048,
                       dur_s: float = 8.0, n_mu: int = 3):
    """Convolutive mixture of ``n_mu`` regular ~12 Hz spike trains."""
    rng = np.random.default_rng(seed)
    T, L = int(dur_s * fs), 32
    emg = np.zeros((n_ch, T))
    gt = {}
    for k in range(n_mu):
        isi = np.abs(rng.normal(fs / 12, fs / 120, size=200))
        t = np.cumsum(isi).astype(int) + rng.integers(0, fs // 12)
        t = t[(t > L) & (t < T - L)]
        gt[k] = t
        s = np.zeros(T)
        s[t] = 1.0
        muap = rng.standard_normal((n_ch, L)) * np.hanning(L)
        for c in range(n_ch):
            emg[c] += np.convolve(s, muap[c], mode="same")
    emg += 0.05 * rng.standard_normal(emg.shape)
    return emg.astype(np.float32), fs, gt


def test_public_api():
    for name in ("run_harmonica", "run_signal_space", "evaluate_decomposition",
                 "preprocess", "run_ica", "refine_harmonica_style"):
        assert hasattr(harmonica, name), name
    assert harmonica.run_harmonica is harmonica.run_signal_space
    import harmonica.baselines as b
    for name in ("run_chen", "run_glaser", "run_glaser_cyclostationary",
                 "run_kramberger", "run_yeung", "run_adapt", "run_tcl", "run_ivae"):
        assert callable(getattr(b, name)), name


def test_power_law_is_sign_preserving():
    """``sign(s) * |s|^a`` for every exponent: troughs never become peaks."""
    np.random.seed(0)
    fs = 2048
    T = 4 * fs
    s = (np.random.randn(T) * 0.2).astype(np.float32)
    peaks = np.arange(100, T - 100, 300)
    troughs = peaks + 150
    s[peaks] = 5.0
    s[troughs] = -5.0
    for a in (2, 2.5, 3, 4):
        sp, sil = est_spike_times(s, fs, a=a)
        assert not np.isnan(sil), a
        assert set(sp.tolist()) == set(peaks.tolist()), a


def test_harmonica_synthetic_mixture():
    emg, fs, gt = _synthetic_mixture()
    spikes = run_harmonica(emg, fs, R=4, d_eff=64, epochs=30, sep_only_warmup=10)
    assert isinstance(spikes, dict) and spikes
    best = _best_f1(spikes, gt, fs)
    assert sum(f >= 0.9 for f in best.values()) >= 2, best


def test_adapt_v1_synthetic_mixture():
    """The v1.0.0 Adapt engine runs from the vendored copy and finds the units."""
    pytest.importorskip("optuna", reason="needs the [adapt] extra")
    from harmonica.baselines import run_adapt_v1
    import adapt_decomp
    vendored = Path(harmonica.__file__).parent / "baselines" / "vendored_adapt_decomp_v1"
    assert Path(adapt_decomp.__file__).resolve().is_relative_to(vendored.resolve())
    emg, fs, gt = _synthetic_mixture()
    spikes = run_adapt_v1(emg, fs, calib_s=2.0, ext_fact=4, d_eff=48, device="cpu")
    assert isinstance(spikes, dict) and spikes
    best = _best_f1(spikes, gt, fs)
    assert sum(f >= 0.9 for f in best.values()) >= 2, best


@pytest.mark.skipif("HARMONICA_DATA" not in os.environ,
                    reason="set HARMONICA_DATA to the benchmark_v2/ directory")
def test_benchmark_recording():
    path = Path(os.environ["HARMONICA_DATA"]) / "ch070" / "SA05-hi_ECRB_sub00_N071.npz"
    d = np.load(path, allow_pickle=True)
    emg = d["emg"].T.astype(np.float32)
    fs = int(d["fs"])
    gt = {int(k.split("_")[-1]): d[k] for k in d.files
          if k.startswith("spike_mu_") and len(d[k]) > 0}
    spikes = run_harmonica(emg, fs, R=8, epochs=200, sep_only_warmup=50,
                           ica_seed=1909, seed=42)
    best = _best_f1(spikes, gt, fs)
    n_good = sum(f >= 0.95 for f in best.values())
    print(f"{n_good} / {len(gt)} units at F1 >= 0.95")
    # The paper's seed-0 run recovered 4 units here; allow one for GPU nondeterminism.
    assert n_good >= 3
