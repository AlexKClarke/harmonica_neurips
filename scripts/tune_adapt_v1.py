#!/usr/bin/env python
"""Validation-split hyperparameter search for the adapt_decomp v1.0.0 baseline.

Tunes ``wh_learning_rate``, ``sv_learning_rate`` and ``centroid_momentum`` for
``adapt_v1`` on the **held-out validation split**, with v1's own optimisers,
and writes a ``best_params.json`` that
``scripts/bench_adapt.py --methods adapt_v1_opt_<approach>_<pool>
--adapt-v1-opt-config`` applies to the test set. The paper's Adapt row uses
``--approach pareto --pool cov --multivariate-tpe --n-startup-trials 30``
(``experiments/tune_adapt.sh``); its outputs are in ``experiments/adapt_tuning/``.

Base config
-----------
Every trial starts from upstream's NeuroMotion adaptation config
(``configs/adapt_configs/default_neuromotion.yaml``, via
:func:`harmonica.baselines.adapt_v1.build_adapt_config`) -- the same
config the vendored ``adapt_v1`` row runs -- and only the three searched
parameters change. Calibration is this repo's shared pipeline
(:func:`calibrate_adapt_v1`), identical to the test-set run.

Approaches (``--approach``), all v1's own ``adapt_decomp.adaptation.optimize``
------------------------------------------------------------------------------
  pareto  optimize_adapt_decomp_pooled_memory_pareto, objectives
          (wh_loss, sv_loss); the config is picked from the front by the
          library default, _select_min_sv_loss.
  sv      optimize_adapt_decomp_pooled_memory, objective="sv_loss".
  roa     optimize_adapt_decomp_pooled_memory, objective="roa"
          (100 - mean rate of agreement with the ground truth, pooled).

Pools (``--pool``) -- which calibration units a search adapts and scores
--------------------------------------------------------------------------
  paired  CBSSResult.select_supervised(roa_th=--roa-th, tol_spike_ms=--roa-tol-ms):
          only units whose calibration spikes match a GT unit with RoA >=
          roa_th (0.8) are kept -- v1's own supervised pooling.
  all     every calibration unit is kept (no quality selection). Each is
          paired to its best GT unit with v1's rate_of_agreement (the same
          greedy 1:1 pairing, no threshold); a unit left unpaired gets an
          all-zero GT column (its RoA is 0 in every trial, so it shifts the
          mean but not the ranking). pareto / sv never read the GT, so on
          this pool they are fully blind.
  sil     units whose silhouette (SIL) on the *un-peeled* calibration source
          (the source the online engine runs) is >= --sil-th (0.9), selected
          with v1's CBSSResult.select_unsupervised -- the quality filter usable
          without ground truth. The ICA's own SIL is measured on the peel-off
          residue and is optimistic for later units, so SIL is recomputed with
          the ICA's est_spike_times on calibration.sources. GT pairing as for
          ``all`` (no threshold), used only for the RoA diagnostic.
  cov     units whose coefficient of variation of the inter-spike intervals
          (CoV-ISI) on the un-peeled calibration source is <= --cov-th (0.3),
          via v1's select_unsupervised(cov_th=...). Spikes are the same
          est_spike_times re-detection as ``sil``; CoV is v1's own
          get_coefficient_of_variation (as its CBSS computes CBSSResult.cov_isi).
          GT pairing as for ``all``.
Every search logs the pooled validation RoA per trial (``compute_roa=True``);
only ``--approach roa`` optimises it.

Search space
------------
v1's DEFAULT_PARAM_SPACE (wh_learning_rate log 1e-4..5e-2, sv_learning_rate
log 1e-4..1e-1) plus ``centroid_momentum`` categorical over 0.1..0.9 (step
0.1). The default sampler is v1's TPESampler(n_startup_trials=15,
seed=--seed); ``--multivariate-tpe`` swaps in the multivariate TPE (joint
Parzen kernels over all three parameters) and ``--n-startup-trials`` changes
the number of random trials before TPE starts. Any non-default sampler is a
separate *variant* (``best_params.json["variant"]``, e.g. ``mvtpe30``) that
bench_adapt.py runs as ``adapt_v1_opt_<approach>_<pool>_<variant>``.

Data split
----------
  benchmark_v2/ch{070,320}      -- 168 recordings/grid, subjects 0-4.
                                  THE TEST SET. Never touched here.
  benchmark_v2_val/ch{070,320}  -- 40 recordings/grid, subjects 5-9
                                  (``recordings_val.tar.gz`` on the dataset
                                  page). Tuned on here: pass this directory
                                  as ``--data-dir``.

Known limitation
----------------
The v1 optimisers create their Optuna study internally with no storage, so a
run is **not resumable** -- a restart begins again at trial 0. ``best_result/``
keeps the best trial's per-recording outputs and ``study.pkl``.

Output (``--out-dir``, by convention ch{g}_v1_{approach}_{pool}[_{variant}])
-----------------------------------------------------------------------------
  best_params.json   -- selected config + provenance, for --adapt-v1-opt-config
  all_trials.csv     -- per-trial losses, pooled RoA and parameters
  pareto_front.csv   -- the final front (pareto only)
  best_result/       -- the optimiser's own best-trial dump + study.pkl
"""
from __future__ import annotations

import os
# Pin BLAS/MKL to one thread before numpy/torch load, so a single-threaded
# trial does not oversubscribe the node.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from harmonica.baselines.adapt import online_start
from harmonica.baselines.adapt_v1 import (
    NEUROMOTION_YAML,
    build_adapt_config,
    calibrate_adapt_v1,
)
from harmonica.ica.detect import est_spike_times
from harmonica.ica.preprocess import bandpass_signals
from bench_adapt import load_recording

from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    optimize_adapt_decomp_pooled_memory,
    optimize_adapt_decomp_pooled_memory_pareto,
)
from adapt_decomp.spikes import rate_of_agreement
from adapt_decomp.spikes.metrics import get_coefficient_of_variation
from adapt_decomp.utils.loaders import PooledDatasetMemory

# Same zero-phase 10-1000 Hz bandpass as bench_adapt.py
_BP_LOW = 10
_BP_HIGH = 1000

# Calibration settings -- must match bench_adapt.py's EXT_FACT / D_EFF /
# CALIB_S / SIL_TH (checked when the benchmark loads best_params.json).
FIXED = dict(
    calib_s=1.5,
    ext_fact=8,
    d_eff=128,
    sil_th=0.8,
    ica_seed=1909,
)

# Search space: v1's DEFAULT_PARAM_SPACE plus the centroid momentum on an
# exact 0.1..0.9 grid (rounded so the choices are exactly 0.1, 0.2, ...).
MOMENTA = [round(0.1 * k, 1) for k in range(1, 10)]
PARAM_SPACE = {
    **DEFAULT_PARAM_SPACE,
    "centroid_momentum": ("categorical", MOMENTA),
}

APPROACHES = {
    # approach: (optimiser, objective(s))
    "pareto": ("optimize_adapt_decomp_pooled_memory_pareto", ("wh_loss", "sv_loss")),
    "sv": ("optimize_adapt_decomp_pooled_memory", "sv_loss"),
    "roa": ("optimize_adapt_decomp_pooled_memory", "roa"),
}
PARAM_NAMES = list(PARAM_SPACE)


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------

def iter_val_recordings(data_dir: Path, ch: int) -> list[Path]:
    """All ``.npz`` paths of the validation split for one grid, sorted.

    Args:
        data_dir: Extracted ``benchmark_v2_val/`` directory.
        ch: Grid, 70 or 320.

    Raises:
        FileNotFoundError: If the split is missing or empty.

    Returns:
        list[Path]: Sorted recording paths.
    """
    rec_dir = data_dir / f"ch{ch:03d}"
    if not rec_dir.exists():
        raise FileNotFoundError(f"Val split not found: {rec_dir}")
    paths = sorted(rec_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {rec_dir}")
    return paths


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _gt_binary(gt_spikes: dict, n_samples: int) -> np.ndarray:
    """``(n_samples, n_gt)`` int8 spike matrix, GT units in sorted-key order."""
    keys = sorted(gt_spikes)
    gt = np.zeros((n_samples, len(keys)), dtype=np.int8)
    for col, k in enumerate(keys):
        idx = np.asarray(gt_spikes[k], dtype=np.int64)
        gt[idx[(idx >= 0) & (idx < n_samples)], col] = 1
    return gt


def unpeeled_quality(calibration, fs: float) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """SIL and CoV-ISI of every un-peeled calibration source.

    Spikes are re-detected on ``calibration.sources`` (the un-peeled sources
    the online engine runs) with the ICA's own ``est_spike_times``, which also
    returns the SIL. The CoV-ISI of those spikes is v1's
    ``get_coefficient_of_variation`` with no ISI discarded, the call v1's CBSS
    uses to fill ``CBSSResult.cov_isi`` (NaN below 3 spikes, which
    ``select_unsupervised(cov_th=...)`` then rejects).

    Args:
        calibration: ``CBSSResult`` from :func:`calibrate_adapt_v1`.
        fs: Sampling frequency, in Hz.

    Returns:
        tuple: ``(spikes, sil, cov_isi)`` -- the re-detected spike indices per
        unit, and two ``(n_mu,)`` float32 arrays.
    """
    T, n_mu = calibration.sources.shape
    sil = np.zeros(n_mu, dtype=np.float32)
    spikes, spikes_bin = [], torch.zeros(T, n_mu, dtype=torch.int32)
    for j in range(n_mu):
        spk, sil[j] = est_spike_times(calibration.sources[:, j], fs)
        spikes.append(np.asarray(spk, dtype=np.int64))
        spikes_bin[spikes[j], j] = 1
    timestamps = torch.arange(T, dtype=torch.float64) / fs
    cov_isi = get_coefficient_of_variation(spikes_bin, timestamps, None).numpy().astype(np.float32)
    return spikes, sil, cov_isi


def build_pool(
    paths: list[Path], pool_kind: str, roa_th: float, roa_tol_ms: float, verbose: bool,
    sil_th: float = 0.9, cov_th: float = 0.3,
) -> tuple[dict[str, PooledDatasetMemory], dict]:
    """Calibrate every validation recording once and wrap it for the optimiser.

    Preprocessing is byte-identical to ``bench_adapt.py``'s: a zero-phase
    10-1000 Hz Butterworth, then :func:`calibrate_adapt_v1` on the first
    ``FIXED["calib_s"]`` seconds, and the online slice the test run feeds the
    engine (starting ``ext_fact`` samples before the end of calibration).

    ``gt_paired_bin`` always has shape ``(T - s_on, M)``, the shape of the
    adapter's spike output, columns in calibration-unit order:

    * ``paired``: the calibration is narrowed with v1's
      ``CBSSResult.select_supervised`` (units with calibration RoA >= roa_th).
    * ``all``: every unit is kept and paired to its best GT unit with v1's
      ``rate_of_agreement`` (no threshold); unpaired units get zero columns.
    * ``sil``: units whose SIL on the un-peeled calibration source is >=
      ``sil_th`` are kept (v1's ``select_unsupervised``), then GT is attached
      as for ``all``.
    * ``cov``: as ``sil``, but keeping units whose un-peeled CoV-ISI is <=
      ``cov_th``.

    Args:
        paths: Validation ``.npz`` paths.
        pool_kind: ``"paired"``, ``"all"``, ``"sil"`` or ``"cov"``.
        roa_th: Calibration RoA threshold for ``paired``.
        roa_tol_ms: Spike tolerance for the calibration pairing, in ms.
        verbose: Print per-recording pairing statistics.
        sil_th: Un-peeled SIL threshold for ``sil``.
        cov_th: Un-peeled CoV-ISI threshold for ``cov``.

    Returns:
        tuple: ``(pool, stats)`` -- pool keyed by recording stem, and a dict
        with ``n_units`` / ``n_units_paired`` / ``skipped``.

    Raises:
        RuntimeError: If no recording yielded a usable pool entry.
    """
    pool: dict[str, PooledDatasetMemory] = {}
    stats = {"n_units": 0, "n_units_paired": 0, "skipped": {}}
    t_start = time.time()

    for i, path in enumerate(paths, 1):
        rec = load_recording(path, drop_empty_gt=False)
        fs = rec["fs"]
        emg = bandpass_signals(rec["emg"], fsamp=fs, high_pass=_BP_LOW, low_pass=_BP_HIGH)
        calibration, cbss_config = calibrate_adapt_v1(
            emg, fs,
            calib_s=FIXED["calib_s"], ext_fact=FIXED["ext_fact"], d_eff=FIXED["d_eff"],
            sil_th=FIXED["sil_th"], ica_seed=FIXED["ica_seed"],
        )
        if calibration is None:
            stats["skipped"][rec["name"]] = "no calibration MUs"
            continue

        T = emg.shape[1]
        T_calib = calibration.emg.shape[0]
        s_on = online_start(T_calib, FIXED["ext_fact"])
        gt = _gt_binary(rec["gt_spikes"], T)
        n_cal = calibration.sources.shape[1]

        if pool_kind == "paired":
            try:
                calibration = calibration.select_supervised(
                    gt[:T_calib], roa_th=roa_th, tol_spike_ms=roa_tol_ms, fs=fs,
                )
            except ValueError:
                stats["skipped"][rec["name"]] = f"no unit with calibration RoA >= {roa_th}"
                continue
            gt_paired = gt[s_on:, calibration.gt_matched_indices]
            roa_cal = np.asarray(calibration.roa, dtype=float)
            n_paired = gt_paired.shape[1]
        else:
            if pool_kind in ("sil", "cov"):
                # Quality of the un-peeled source (calibration.sources, what
                # the engine runs), swapped into a copy so v1's own
                # select_unsupervised applies the threshold.
                _, sil_unpeeled, cov_unpeeled = unpeeled_quality(calibration, fs)
                crit = {"sil": ("sil_th", sil_th, ">="), "cov": ("cov_th", cov_th, "<=")}[pool_kind]
                try:
                    calibration = dataclasses.replace(calibration, sil=sil_unpeeled, cov_isi=cov_unpeeled) \
                        .select_unsupervised(**{crit[0]: crit[1]})
                except ValueError:
                    stats["skipped"][rec["name"]] = f"no unit with un-peeled {pool_kind.upper()} {crit[2]} {crit[1]}"
                    continue
            n_keep = calibration.sources.shape[1]
            roa_vals, pairs, _ = rate_of_agreement(
                gt[:T_calib].astype(np.float32), calibration.spikes.astype(np.float32),
                fs=int(fs), tol_spike_ms=roa_tol_ms,
            )
            gt_paired = np.zeros((T - s_on, n_keep), dtype=np.int8)
            roa_cal = np.zeros(n_keep)
            for (gt_idx, dec_idx), r in zip(pairs, roa_vals):
                gt_paired[:, dec_idx] = gt[s_on:, gt_idx]
                roa_cal[dec_idx] = r
            n_paired = len(pairs)

        n_units = calibration.sources.shape[1]
        assert gt_paired.shape == (T - s_on, n_units), (gt_paired.shape, T - s_on, n_units)
        pool[rec["name"]] = PooledDatasetMemory(
            emg=torch.from_numpy(np.ascontiguousarray(emg[:, s_on:].T, dtype=np.float32)),
            calibration=calibration,
            cbss_config=cbss_config,
            preprocess=False,       # already bandpassed above
            gt_paired_bin=gt_paired,
        )
        stats["n_units"] += n_units
        stats["n_units_paired"] += n_paired
        if verbose:
            print(f"  [{i:3d}/{len(paths)}] {rec['name']:<32s} calib MUs={n_cal:3d} "
                  f"kept={n_units:3d} paired={n_paired:3d} "
                  f"median calib RoA={np.median(roa_cal) if len(roa_cal) else float('nan'):.3f}",
                  flush=True)

    if not pool:
        raise RuntimeError("No validation recording produced a usable pool entry.")
    print(f"Pool ({pool_kind}): {len(pool)} recordings, {stats['n_units']} units "
          f"({stats['n_units_paired']} GT-paired) in {time.time() - t_start:.0f}s"
          + (f"; skipped {len(stats['skipped'])}: {stats['skipped']}" if stats["skipped"] else ""))
    return pool, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via a temp file + rename, so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def make_trial_logger(out_dir: Path, verbose: bool):
    """``on_trial`` callback appending one row per trial to ``all_trials.csv``.

    Works for both optimiser families: the single-objective log has a
    ``loss`` key, the Pareto log a ``values`` tuple; both carry the pooled
    ``sv_loss`` / ``wh_loss`` / ``total_loss``, ``params`` and (with
    ``compute_roa``) ``roa_mean``.

    Args:
        out_dir: Output directory.
        verbose: Also print each trial.

    Returns:
        Callable[[dict], None]: Callback for the optimiser's ``on_trial``.
    """
    csv_path = out_dir / "all_trials.csv"
    header = ["trial", "objective_value", "wh_loss", "sv_loss", "total_loss",
              "roa_mean_pooled", *PARAM_NAMES]
    state = {"t0": time.time()}

    def _on_trial(log: dict) -> None:
        obj = log.get("loss", log.get("values"))
        row = [log["trial_number"], obj, log.get("wh_loss"), log.get("sv_loss"),
               log.get("total_loss"), log.get("roa_mean"),
               *(log["params"].get(k) for k in PARAM_NAMES)]
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)
        if verbose:
            p = log["params"]
            print(f"  trial {log['trial_number']:3d}  obj={obj}  "
                  f"wh={log.get('wh_loss'):.4g} sv={log.get('sv_loss'):.4g} "
                  f"roa={log.get('roa_mean', float('nan')):.2f}  "
                  f"wh_lr={p.get('wh_learning_rate'):.3e} sv_lr={p.get('sv_learning_rate'):.3e} "
                  f"mom={p.get('centroid_momentum')}  [{(time.time() - state['t0']) / 60:.1f} min]",
                  flush=True)

    return _on_trial


def write_front(out_dir: Path, front) -> None:
    """Dump the final Pareto front to ``pareto_front.csv``."""
    with (out_dir / "pareto_front.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "wh_loss", "sv_loss", "total_loss", "roa_mean_pooled", *PARAM_NAMES])
        for t in sorted(front, key=lambda x: x.number):
            w.writerow([t.number, t.user_attrs.get("wh_loss"), t.user_attrs.get("sv_loss"),
                        t.user_attrs.get("total_loss"), t.user_attrs.get("roa_mean_pooled"),
                        *(t.params.get(k) for k in PARAM_NAMES)])


def _to_jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Extracted benchmark_v2_val/ directory (contains ch070/ and ch320/).")
    p.add_argument("--ch", type=int, default=320, choices=[70, 320])
    p.add_argument("--approach", required=True, choices=list(APPROACHES))
    p.add_argument("--pool", required=True, choices=["paired", "all", "sil", "cov"])
    p.add_argument("--roa-th", type=float, default=0.8,
                   help="Calibration RoA threshold for --pool paired (select_supervised).")
    p.add_argument("--sil-th", type=float, default=0.9,
                   help="Un-peeled calibration SIL threshold for --pool sil (select_unsupervised).")
    p.add_argument("--cov-th", type=float, default=0.3,
                   help="Un-peeled calibration CoV-ISI threshold (keep <=) for --pool cov "
                        "(select_unsupervised).")
    p.add_argument("--roa-tol-ms", type=float, default=2.0,
                   help="Spike tolerance (ms) for the GT pairing and the RoA objective/diagnostic "
                        "-- 2 ms, as in the benchmark's matching.")
    p.add_argument("--n-trials", type=int, default=100,
                   help="Optuna trial budget (v1's default). Each trial runs the whole pool.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Optuna trial-level parallelism. Keep at 1 for exact seed reproducibility.")
    p.add_argument("--multivariate-tpe", action="store_true",
                   help="Use TPESampler(multivariate=True) instead of v1's default independent TPE.")
    p.add_argument("--n-startup-trials", type=int, default=15,
                   help="Random trials before TPE starts (v1's default: 15).")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1909, help="TPESampler seed (v1's own default).")
    p.add_argument("--max-recordings", type=int, default=None,
                   help="Truncate the val pool -- smoke tests only.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--no-best-result-path", action="store_true",
                   help="Skip the optimiser's best-trial dump and study.pkl checkpointing.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    """Build the pool, run the selected v1 search, and write the chosen config.

    Returns:
        int: Process exit status.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fn_name, objective = APPROACHES[args.approach]
    # None -> the optimiser builds v1's default TPESampler(n_startup_trials=15,
    # seed=random_seed). Anything else is a named variant of the search.
    default_sampler = not args.multivariate_tpe and args.n_startup_trials == 15
    sampler = None if default_sampler else optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup_trials, seed=args.seed, multivariate=args.multivariate_tpe)
    sampler_desc = ("TPESampler(n_startup_trials=15) [v1 default]" if default_sampler else
                    f"TPESampler(n_startup_trials={args.n_startup_trials}"
                    + (", multivariate=True)" if args.multivariate_tpe else ")"))
    variant = "" if default_sampler else \
        f"{'mvtpe' if args.multivariate_tpe else 'tpe'}{args.n_startup_trials}"

    paths = iter_val_recordings(args.data_dir, args.ch)
    if args.max_recordings:
        paths = paths[: args.max_recordings]

    print(f"adapt_v1 tuning: grid=ch{args.ch} approach={args.approach} ({fn_name}, {objective}) "
          f"pool={args.pool} recordings={len(paths)} trials={args.n_trials} n_jobs={args.n_jobs}\n"
          f"  base config={NEUROMOTION_YAML.name}  fixed={FIXED}\n"
          f"  param_space={PARAM_SPACE}\n  sampler={sampler_desc}\n  out={args.out_dir}\n", flush=True)

    pool, pool_stats = build_pool(paths, args.pool, args.roa_th, args.roa_tol_ms, args.verbose,
                                  sil_th=args.sil_th, cov_th=args.cov_th)

    # One representative cbss_config seeds the base AdaptConfig; every entry's
    # config is identical by construction (same ext_fact/ext_mode/fs/filters).
    any_cfg = next(iter(pool.values())).cbss_config
    base_config = build_adapt_config(any_cfg, compute_loss=True, device=args.device)
    best_result_path = None if args.no_best_result_path else str(args.out_dir / "best_result")
    common = dict(
        pool=pool, base_config=base_config, compute_roa=True,
        roa_kwargs={"tol_spike_ms": args.roa_tol_ms}, param_space=PARAM_SPACE,
        n_trials=args.n_trials, n_jobs=args.n_jobs, sampler=sampler, random_seed=args.seed,
        best_result_path=best_result_path, on_trial=make_trial_logger(args.out_dir, args.verbose),
    )

    t0 = time.time()
    front = None
    if args.approach == "pareto":
        # selection_rule=None -> _select_min_sv_loss, the library default.
        returned = optimize_adapt_decomp_pooled_memory_pareto(objectives=objective, selection_rule=None, **common)
        # (best_config, front, study), or (best_outputs, best_config, front, study)
        # with best_result_path set.
        best_config, front, study = returned[-3:]
        best_trial = min(front, key=lambda t: t.user_attrs["sv_loss"])
        write_front(args.out_dir, front)
        selection_rule = "_select_min_sv_loss (library default)"
    else:
        returned = optimize_adapt_decomp_pooled_memory(objective=objective, **common)
        # (best_config, study), or (best_outputs, best_config, study) with best_result_path set.
        best_config, study = returned[-2:]
        best_trial = study.best_trial
        selection_rule = f"min {objective} (study.best_trial)"
    elapsed = time.time() - t0

    ua = best_trial.user_attrs
    payload = {
        **FIXED,
        "grid": args.ch,
        "approach": args.approach,
        "pool": args.pool,
        "optimiser": fn_name,
        "objective": list(objective) if isinstance(objective, tuple) else objective,
        "selection_rule": selection_rule,
        # --- the selected parameters ---
        "wh_learning_rate": float(best_config.wh_learning_rate),
        "sv_learning_rate": float(best_config.sv_learning_rate),
        "centroid_momentum": float(best_config.centroid_momentum),
        # --- provenance ---
        "base_config": NEUROMOTION_YAML.name,
        "param_space": {k: _to_jsonable(v) for k, v in PARAM_SPACE.items()},
        "sampler": sampler_desc,
        "n_startup_trials": args.n_startup_trials,
        "multivariate_tpe": args.multivariate_tpe,
        # "" for v1's default sampler; else the method suffix bench_adapt.py expects
        "variant": variant,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "roa_th": args.roa_th if args.pool == "paired" else None,
        # Not "sil_th": that key (from FIXED) is the calibration ICA's own
        # threshold, which bench_adapt.py checks against SIL_TH.
        "pool_sil_th": args.sil_th if args.pool == "sil" else None,
        "pool_cov_th": args.cov_th if args.pool == "cov" else None,
        "roa_tol_ms": args.roa_tol_ms,
        "library": "adapt_decomp",
        "library_version": "1.0.0",
        "trial_number": best_trial.number,
        "pooled_wh_loss": float(ua.get("wh_loss", float("nan"))),
        "pooled_sv_loss": float(ua.get("sv_loss", float("nan"))),
        "pooled_total_loss": float(ua.get("total_loss", float("nan"))),
        "roa_mean_pooled": float(ua.get("roa_mean_pooled", float("nan"))),
        "n_front": len(front) if front is not None else None,
        "n_pool": len(pool),
        "n_units": pool_stats["n_units"],
        "n_units_paired": pool_stats["n_units_paired"],
        "skipped": pool_stats["skipped"],
    }
    _write_json_atomic(args.out_dir / "best_params.json", payload)

    print(f"\nDone in {elapsed / 60:.1f} min; selected trial {best_trial.number}:\n"
          f"  wh_learning_rate  = {payload['wh_learning_rate']:.6e}\n"
          f"  sv_learning_rate  = {payload['sv_learning_rate']:.6e}\n"
          f"  centroid_momentum = {payload['centroid_momentum']}\n"
          f"  pooled wh_loss={payload['pooled_wh_loss']:.6g} sv_loss={payload['pooled_sv_loss']:.6g} "
          f"roa_mean={payload['roa_mean_pooled']:.2f}\n"
          f"  -> {args.out_dir / 'best_params.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
