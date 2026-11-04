#!/usr/bin/env python
"""Adapt (Mendez Guerra 2024) benchmark: the Adapt rows and their ICA reference.

Runs the Adapt wrappers on every recording of one or both grids with the
paper's five FastICA seeds and writes one long-format row per
``(grid, recording, seed, method, GT unit)``: that unit's best-match F1 and
the recall of the same match.

Methods:
    adapt_v1_opt_<approach>_<pool>[_<variant>]
        adapt_decomp v1.0.0 with ``(wh_learning_rate, sv_learning_rate,
        centroid_momentum)`` from a ``scripts/tune_adapt_v1.py``
        ``best_params.json`` (``--adapt-v1-opt-config``). The paper's Adapt row
        is ``adapt_v1_opt_pareto_cov_mvtpe30`` with the configs in
        ``experiments/adapt_tuning/``.
    adapt_v1        v1.0.0 engine with upstream's NeuroMotion config.
    adapt           v0 engine (``vendored_adapt_decomp/``) with its defaults.
    ica_cbss        the unrefined FastICA pool HarmonICA warm-starts from.

Every method sees the same input: a zero-phase 10-1000 Hz band-pass of the
whole recording, then the shared Toeplitz (L=8) -> PCA (128) -> ZCA ->
FastICA (SIL >= 0.8) calibration on the first 1.5 s. HarmonICA and the other
baselines in the paper's tables come from ``bench_baselines.py`` /
``bench_sweep.py``, which do not band-pass.

Ground-truth units with an empty spike train are dropped on load, as in the
other runners; ``n_gt`` and the tiers count the remaining units. An empty
train can never be an estimate's strict best match (its F1 is 0), so this
leaves every other unit's ``best_f1`` / ``best_recall`` unchanged.

Example:
    python scripts/bench_adapt.py --data-dir /path/to/benchmark_v2 --ch 70 \\
        --device cpu --methods adapt_v1_opt_pareto_cov_mvtpe30 \\
        --adapt-v1-opt-config experiments/adapt_tuning/ch70_v1_pareto_cov_mvtpe30/best_params.json \\
        --out results/ch070_adapt_tuned.csv

    # one shard of 16 (recordings split into contiguous chunks):
    python scripts/bench_adapt.py ... --num-shards 16 --shard-idx 3 --out shard3.csv
"""
from __future__ import annotations

import os
# Pin BLAS threads before numpy/sklearn load (one single-threaded shard per core).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from harmonica.baselines import run_adapt, run_adapt_v1
from harmonica.evaluate import evaluate_decomposition
from harmonica.ica.algorithm import run_ica_loop
from harmonica.ica.preprocess import bandpass_signals
from harmonica.ica.utils import extension
from harmonica.ica.whitening import whitening as _whiten


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

def iter_recordings(data_dir: Path, ch: int) -> Iterator[Path]:
    """Sorted ``.npz`` paths of one grid under ``data_dir`` (``benchmark_v2/``)."""
    rec_dir = data_dir / f"ch{ch:03d}"
    if not rec_dir.exists():
        raise FileNotFoundError(f"Recording directory not found: {rec_dir}")
    paths = sorted(rec_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {rec_dir}")
    return iter(paths)


def load_recording(npz_path: Path, drop_empty_gt: bool = True) -> dict:
    """Load one recording, EMG channels-first.

    Args:
        npz_path: Benchmark ``.npz`` (EMG stored samples-first).
        drop_empty_gt: Drop ground-truth units with no spikes.

    Returns:
        Dict with ``emg`` ``(C, T)`` float32, ``fs``, ``angle_profile``,
        ``gt_spikes`` ``{mu_id: spike sample indices}`` and ``name``.
    """
    d = np.load(npz_path, allow_pickle=False)
    emg = d["emg"].T.astype(np.float32)          # (T, C) -> (C, T)
    gt_spikes = {
        int(k.split("_")[-1]): d[k].astype(np.int64)
        for k in d.files
        if k.startswith("spike_mu_") and (len(d[k]) > 0 or not drop_empty_gt)
    }
    return {
        "emg":           emg,
        "fs":            int(d["fs"]),
        "angle_profile": d["angle_profile"].astype(np.float32),
        "gt_spikes":     gt_spikes,
        "name":          npz_path.stem,
    }


# ---------------------------------------------------------------------------
# Shared settings (paper Table 13)
# ---------------------------------------------------------------------------
BANDPASS_LOW_HZ  = 10
BANDPASS_HIGH_HZ = 1000

EXT_FACT = 8       # L -- uniform across every baseline, including Adapt.
D_EFF    = 128
SIL_TH   = 0.80
CALIB_S  = 1.5     # uniform across every stochastic baseline, including Adapt.

# Adapt v0: the vendored Config defaults (NeuroMotion: Table 15 learning rates).
ADAPT_WH_LR = 7e-3
ADAPT_SV_LR = 3e-3

# adapt_v1 takes every adaptation setting from upstream's
# default_neuromotion.yaml via run_adapt_v1's defaults -- nothing to set here.
# adapt_v1_opt_<approach>_<pool> overrides three of them with the values
# scripts/tune_adapt_v1.py selected on the validation split (loaded in main()
# from --adapt-v1-opt-config; one method + grid per call).
# (approach, pool) pairs that have a validation search: every approach on the
# GT-paired and all-units pools; pareto / sv also on the un-peeled SIL >= 0.9
# and un-peeled CoV-ISI <= 0.3 pools. A third element names a non-default
# sampler variant of a search (tune_adapt_v1.py's best_params.json "variant";
# "" = v1's default TPESampler): mvtpe30 = multivariate TPE, 30 startup trials.
ADAPT_V1_OPT_SEARCHES = (
    ("pareto", "paired"), ("pareto", "all"), ("pareto", "sil"), ("pareto", "cov"),
    ("sv", "paired"), ("sv", "all"), ("sv", "sil"), ("sv", "cov"),
    ("roa", "paired"), ("roa", "all"),
    ("pareto", "cov", "mvtpe30"), ("sv", "cov", "mvtpe30"),
)
ADAPT_V1_OPT_PARAMS = ("wh_learning_rate", "sv_learning_rate", "centroid_momentum")
ADAPT_V1_OPT: dict | None = None
# Adapt torch device override -- None means auto-detect (cuda if available).
# A single-recording timing check found CPU at least as fast as GPU for
# Adapt's small (100ms-batch) online updates, so --device cpu is a fine,
# GPU-free choice; set via main()'s --device flag.
ADAPT_DEVICE: str | None = None

# 5-seed paper convention (ica_seed, nn_seed) pairs.
SEED_PAIRS = [(1909 + i, 42 + i) for i in range(5)]

MATCH_TOL_S = 0.002
MAX_SHIFT_S = 0.025

# The paper's pool-size tiers
def ngt_tier(n_gt: int) -> str:
    if n_gt < 40:
        return "<40"
    if n_gt < 70:
        return "40-69"
    return ">=70"


# ---------------------------------------------------------------------------
# "ICA / cBSS" reference -- the unrefined FastICA seed pool HarmonICA warm-
# starts from, scored before the compensator acts. Mirrors the
# preprocessing + run_ica_loop call in baselines/signal_space.py.
# ---------------------------------------------------------------------------

def run_ica_reference(
    emg: np.ndarray,
    fs: float,
    R: int = EXT_FACT,
    d_eff: int = D_EFF,
    sil_th: float = SIL_TH,
    ica_seed: int = 1909,
) -> dict[int, np.ndarray]:
    """The unrefined FastICA seed pool, scored as its own method.

    Same Toeplitz(L) -> PCA(K) -> ZCA -> FastICA(peel_off=False) pipeline
    `run_signal_space` builds internally as HarmonICA's warm-start seed;
    returned directly here (instead of being discarded after `filters` is
    read off) so it can be scored on the same band-passed input as Adapt.
    """
    C, _T = emg.shape
    z_ext = extension(emg, R)
    d = min(d_eff, z_ext.shape[0] - 1)
    from sklearn.decomposition import PCA
    # random_state pins sklearn's randomized SVD solver, which 'auto' selects
    # at these shapes; without it the reduction -- and so the whole calibration
    # -- is not reproducible even for a fixed ICA seed.
    z_pca = PCA(n_components=d, random_state=ica_seed).fit_transform(z_ext.T).T
    z_w, _ = _whiten(z_pca, method="ZCA")

    n_iter = min(C * 3, 108)
    _src, spikes, _sil, filters = run_ica_loop(
        z_w, fs, sil_th=sil_th, n_iter=n_iter,
        seed=ica_seed, refinement_loop=True, peel_off=False,
    )
    if filters.shape[1] == 0:
        return {}
    return spikes


# ---------------------------------------------------------------------------
# Method dispatch -- every entry takes (emg_bp, fs, angle, ica_seed, nn_seed)
# and returns {mu_id: spike_indices}.
# ---------------------------------------------------------------------------

def _ica_cbss(emg, fs, angle, ica_seed, nn_seed, calib_s):
    return run_ica_reference(emg, fs, R=EXT_FACT, d_eff=D_EFF, sil_th=SIL_TH, ica_seed=ica_seed)

def _adapt_impl(emg, fs, angle, ica_seed, nn_seed, calib_s, wh_lr, sv_lr):
    return run_adapt(
        emg, fs, calib_s=calib_s, ext_fact=EXT_FACT, d_eff=D_EFF, sil_th=SIL_TH,
        ica_seed=ica_seed, wh_learning_rate=wh_lr, sv_learning_rate=sv_lr,
        bandpass=False,  # already filtered 10-1000 Hz in run_recording
        device=ADAPT_DEVICE,
    )

def _adapt(emg, fs, angle, ica_seed, nn_seed, calib_s):
    return _adapt_impl(emg, fs, angle, ica_seed, nn_seed, calib_s, ADAPT_WH_LR, ADAPT_SV_LR)

def _adapt_v1_impl(emg, fs, angle, ica_seed, nn_seed, calib_s, **overrides):
    # Same shared calibration and ext_fact/d_eff/sil_th/calib_s as _adapt_impl;
    # the online engine is v1 with upstream's NeuroMotion config, plus any
    # tuned overrides.
    return run_adapt_v1(
        emg, fs, calib_s=calib_s, ext_fact=EXT_FACT, d_eff=D_EFF, sil_th=SIL_TH,
        ica_seed=ica_seed,
        bandpass=False,  # already filtered 10-1000 Hz in run_recording
        device=ADAPT_DEVICE,
        **overrides,
    )

def _adapt_v1(emg, fs, angle, ica_seed, nn_seed, calib_s):
    return _adapt_v1_impl(emg, fs, angle, ica_seed, nn_seed, calib_s)

def _adapt_v1_opt(emg, fs, angle, ica_seed, nn_seed, calib_s):
    return _adapt_v1_impl(emg, fs, angle, ica_seed, nn_seed, calib_s, **ADAPT_V1_OPT)

# method name -> (approach, pool, variant)
ADAPT_V1_OPT_SPEC = {
    "_".join(["adapt_v1_opt", *s]): (s[0], s[1], s[2] if len(s) > 2 else "")
    for s in ADAPT_V1_OPT_SEARCHES
}
ADAPT_V1_OPT_METHODS = list(ADAPT_V1_OPT_SPEC)

# name -> (dispatch fn, seeded?); every method here uses all 5 seeds.
METHODS: dict[str, tuple[callable, bool]] = {
    "ica_cbss":   (_ica_cbss,   True),
    "adapt":      (_adapt,      True),
    "adapt_v1":   (_adapt_v1,   True),
    # All share one dispatch fn; main() loads the one config per call.
    **{m: (_adapt_v1_opt, True) for m in ADAPT_V1_OPT_METHODS},
}


# ---------------------------------------------------------------------------
# Per-recording, per-method, per-seed evaluation -> long-format rows
# ---------------------------------------------------------------------------

def eval_one(
    fn: callable, emg_bp: np.ndarray, fs: int, angle: np.ndarray,
    gt: dict, ica_seed: int, nn_seed: int, calib_s: float, verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
    """Run one method once; return (gt_best_f1, gt_best_recall, n_est, elapsed, ok)."""
    t0 = time.time()
    try:
        spikes = fn(emg_bp, fs, angle, ica_seed, nn_seed, calib_s)
    except Exception:
        if verbose:
            traceback.print_exc(limit=3)
        return np.zeros(len(gt)), np.zeros(len(gt)), 0, time.time() - t0, False
    elapsed = time.time() - t0
    if not spikes:
        return np.zeros(len(gt)), np.zeros(len(gt)), 0, elapsed, True
    result = evaluate_decomposition(spikes, gt, fsamp=fs, tol=MATCH_TOL_S, max_shift=MAX_SHIFT_S)
    return result["gt_best_f1"], result["gt_best_recall"], len(spikes), elapsed, True


def run_recording(
    rec: dict, ch: int, methods: dict, verbose: bool, calib_s: float = CALIB_S
) -> list[dict]:
    emg_bp = bandpass_signals(
        rec["emg"], fsamp=rec["fs"], high_pass=BANDPASS_LOW_HZ, low_pass=BANDPASS_HIGH_HZ
    )
    fs, angle, gt = rec["fs"], rec["angle_profile"], rec["gt_spikes"]
    n_gt = len(gt)
    tier = ngt_tier(n_gt)
    gt_ids_sorted = sorted(gt.keys())

    rows: list[dict] = []
    for method_name, (fn, is_seeded) in methods.items():
        seed_pairs = SEED_PAIRS if is_seeded else SEED_PAIRS[:1]
        for seed_idx, (ica_seed, nn_seed) in enumerate(seed_pairs):
            gt_best_f1, gt_best_recall, n_est, elapsed, ok = eval_one(
                fn, emg_bp, fs, angle, gt, ica_seed, nn_seed, calib_s, verbose
            )
            if verbose:
                print(
                    f"    [{method_name:<10s}] seed={ica_seed} "
                    f"n_est={n_est} yield@0.95={int((gt_best_f1 >= 0.95).sum())} "
                    f"({elapsed:.1f}s){'  FAILED' if not ok else ''}",
                    flush=True,
                )
            for gt_local_idx, gt_id in enumerate(gt_ids_sorted):
                rows.append({
                    "grid": ch,
                    "recording": rec["name"],
                    "n_gt": n_gt,
                    "ngt_tier": tier,
                    "seed_idx": seed_idx,
                    "ica_seed": ica_seed,
                    "method": method_name,
                    "gt_id": gt_id,
                    "best_f1": float(gt_best_f1[gt_local_idx]),
                    "best_recall": float(gt_best_recall[gt_local_idx]),
                    "n_est": n_est,
                    "time_s": elapsed,
                    "ok": ok,
                })
    return rows


# ---------------------------------------------------------------------------
# HPC sharding -- split a grid's sorted recording list into contiguous chunks.
# Both grids share identical recording filenames (paper §B: same pools, same
# 8 angle/SNR conditions rendered on both electrode configs), so the same
# shard_idx selects the same recordings on every grid -- verified below rather
# than assumed.
# ---------------------------------------------------------------------------

def shard_paths(paths: list[Path], num_shards: int, shard_idx: int) -> list[Path]:
    if not (0 <= shard_idx < num_shards):
        raise SystemExit(f"--shard-idx must be in [0, {num_shards}), got {shard_idx}")
    return list(np.array_split(np.array(paths, dtype=object), num_shards)[shard_idx])


def check_shards_consistent(data_dir: Path, chs: list[int], num_shards: int, shard_idx: int) -> None:
    """When >1 grid is requested, confirm shard_idx picks the same recording
    names on every grid -- the assumption --shard-idx's docstring promises."""
    if len(chs) < 2 or num_shards <= 1:
        return
    stems_by_ch = {}
    for ch in chs:
        paths = shard_paths(list(iter_recordings(data_dir, ch)), num_shards, shard_idx)
        stems_by_ch[ch] = sorted(p.stem for p in paths)
    ref_ch, ref_stems = chs[0], stems_by_ch[chs[0]]
    for ch in chs[1:]:
        if stems_by_ch[ch] != ref_stems:
            raise SystemExit(
                f"Shard {shard_idx}/{num_shards} selects different recordings on "
                f"ch{ref_ch:03d} vs ch{ch:03d} -- the grids' recording lists are "
                "not aligned the way --shard-idx assumes. Run each grid as a "
                "separate, single---ch invocation instead."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Extracted benchmark_v2/ directory (contains ch070/ and ch320/).")
    p.add_argument("--ch", type=int, nargs="+", choices=[70, 320], default=[320, 70],
                   help="Electrode grid(s) to benchmark (default: both).")
    p.add_argument("--max-recordings", type=int, default=None,
                   help="Limit to the first N recordings per grid, taken AFTER "
                        "sharding (smoke tests).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split each grid's sorted recording list into this many "
                        "contiguous shards, for HPC array jobs. "
                        "Default: 1 (no sharding).")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="Which shard to run, 0-indexed, < --num-shards. The same "
                        "index always selects the same recordings regardless of "
                        "--methods/--ch (checked at startup across grids).")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated ica_seed subset (e.g. '1909,1910') to "
                        "shrink the seed loop for smoke tests. Default: all 5.")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated method subset (e.g. 'adapt_v1,ica_cbss') "
                        f"from {sorted(METHODS)}. Default: all.")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="Force the torch device for Adapt's online-adaptation "
                        "engine (the only method here that uses torch/GPU by "
                        "default). Default: auto-detect cuda if available. A "
                        "single-recording timing check found CPU at least as "
                        "fast as GPU for Adapt's small 100ms-batch updates, so "
                        "--device cpu is a fine choice for GPU-free nodes.")
    p.add_argument("--adapt-v1-opt-config", type=Path, default=None,
                   help="scripts/tune_adapt_v1.py best_params.json, required when "
                        "--methods includes an adapt_v1_opt_<approach>_<pool> method. "
                        "Only one such method and one --ch grid per call: the "
                        "config's approach/pool/grid and its calibration settings "
                        "(ext_fact/d_eff/calib_s/sil_th, NeuroMotion base config) "
                        "are checked against this run at startup.")
    p.add_argument("--out", type=Path, required=True,
                   help="Long-format CSV output path. Rows are appended "
                        "incrementally (one flush per recording) so a crash "
                        "mid-sweep doesn't lose prior progress.")
    p.add_argument("--calib_s", type=float, default=CALIB_S,
                   help="Override the default 1.5s calibration window")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_adapt_v1_opt_config(method: str, config_path: Path | None, grids: list[int]) -> dict:
    """Load and validate the tune_adapt_v1.py best_params.json for one opt method.

    Exits loudly on a missing flag or any mismatch (calibration settings, base
    config, approach/pool, grid), so a method never silently runs under a
    config that isn't comparable to the rest of the table.

    Args:
        method: ``adapt_v1_opt_<approach>_<pool>[_<variant>]``.
        config_path: ``--adapt-v1-opt-config``.
        grids: ``--ch`` (must be the single grid the config was tuned on).

    Returns:
        dict: ``{wh_learning_rate, sv_learning_rate, centroid_momentum}`` --
        the ``run_adapt_v1`` overrides.
    """
    if config_path is None:
        raise SystemExit(f"--adapt-v1-opt-config is required when --methods includes {method}")
    cfg = json.loads(config_path.read_text())
    approach, pool, variant = ADAPT_V1_OPT_SPEC[method]
    expected = [
        ("ext_fact", EXT_FACT), ("d_eff", D_EFF), ("calib_s", CALIB_S), ("sil_th", SIL_TH),
        ("base_config", "default_neuromotion.yaml"),
        ("approach", approach), ("pool", pool), ("grid", grids[0] if len(grids) == 1 else grids),
        ("variant", variant),
    ]
    # configs written before variants existed have no "variant" key: v1's default sampler
    mismatches = [f"{k}: config={cfg.get(k, '' if k == 'variant' else None)!r} != expected={v!r}"
                  for k, v in expected if cfg.get(k, "" if k == "variant" else None) != v]
    missing = [k for k in ADAPT_V1_OPT_PARAMS if k not in cfg]
    if mismatches or missing:
        raise SystemExit(
            f"{config_path} does not fit {method} on --ch {grids}:\n  "
            + "\n  ".join(mismatches + [f"missing {k}" for k in missing])
        )
    return {k: cfg[k] for k in ADAPT_V1_OPT_PARAMS}


def main() -> int:
    args = parse_args()

    global ADAPT_DEVICE
    ADAPT_DEVICE = args.device

    methods = dict(METHODS)
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        unknown = keep - set(METHODS)
        if unknown:
            raise SystemExit(f"Unknown method(s): {unknown}. Choices: {sorted(METHODS)}")
        methods = {k: v for k, v in METHODS.items() if k in keep}

    opt_methods = [m for m in methods if m in ADAPT_V1_OPT_METHODS]
    if len(opt_methods) > 1:
        raise SystemExit(f"Run one adapt_v1_opt_* method per call (got {opt_methods}): "
                         "each has its own --adapt-v1-opt-config.")
    if opt_methods:
        if len(args.ch) != 1:
            raise SystemExit(f"{opt_methods[0]} needs a single --ch grid (its config is grid-specific).")
        global ADAPT_V1_OPT
        ADAPT_V1_OPT = _load_adapt_v1_opt_config(opt_methods[0], args.adapt_v1_opt_config, args.ch)

    global SEED_PAIRS
    if args.seeds:
        wanted = {int(s) for s in args.seeds.split(",")}
        restricted = [p for p in SEED_PAIRS if p[0] in wanted]
        if not restricted:
            raise SystemExit(f"--seeds {args.seeds} matched none of {SEED_PAIRS}")
        SEED_PAIRS = restricted

    if args.num_shards > 1:
        check_shards_consistent(args.data_dir, args.ch, args.num_shards, args.shard_idx)

    _uses_adapt = any(m.startswith("adapt") for m in methods)
    print(
        f"Adapt benchmark: grids={args.ch}  methods={sorted(methods)}  "
        f"seeds={SEED_PAIRS}\n  ext_fact={EXT_FACT}  d_eff={D_EFF}  "
        f"sil_th={SIL_TH}  calib_s={args.calib_s}"
        + (f"  adapt_device={args.device or 'auto'}" if _uses_adapt else "")
        + (f"\n  {opt_methods[0]}: {ADAPT_V1_OPT} (from {args.adapt_v1_opt_config})"
           if opt_methods else "")
        + (f"\n  shard={args.shard_idx}/{args.num_shards}" if args.num_shards > 1 else "")
        + f"\n  out={args.out}\n"
    )

    first_write = not args.out.exists()
    total_rows = 0
    for ch in args.ch:
        paths = list(iter_recordings(args.data_dir, ch))
        if args.num_shards > 1:
            paths = shard_paths(paths, args.num_shards, args.shard_idx)
        if args.max_recordings is not None:
            paths = paths[: args.max_recordings]
        shard_note = f" (shard {args.shard_idx}/{args.num_shards})" if args.num_shards > 1 else ""
        print(f"== ch{ch:03d}: {len(paths)} recordings{shard_note} ==")
        for i, path in enumerate(paths):
            rec = load_recording(path)
            print(f"[{i+1:3d}/{len(paths)}] ch{ch:03d} {rec['name']} "
                  f"n_gt={len(rec['gt_spikes'])}", flush=True)
            rows = run_recording(rec, ch, methods, args.verbose)
            df = pd.DataFrame(rows)
            df.to_csv(args.out, mode="a", header=first_write, index=False)
            first_write = False
            total_rows += len(rows)

    print(f"\nDone. {total_rows} long-format rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
