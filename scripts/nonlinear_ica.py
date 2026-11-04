#!/usr/bin/env python
"""Nonlinear-ICA comparison: linear ICA against TCL and iVAE.

Runs each method on a short list of recordings and scores it by the number of
ground-truth units recovered at F1 > 0.5 (best match per unit) and the mean F1
of those units. Every method sees the same whitened input: Toeplitz extension
(R=4), PCA to 128, ZCA (:func:`harmonica.nonstationary.pipelines.preprocess`).

Protocols (``--protocol``), each run per recording in the order listed:

  angle  linear ICA, the FastICA reference (:func:`run_ica`); TCL-θ, the
         classifier trained on 24 quantile bins of the joint angle; iVAE-θ, the
         same bins as the prior's auxiliary variable. TCL prints its training
         progress, which puts the classifier in eval mode from epoch 51 on
         (:meth:`harmonica.models.tcl.DeepTCL.fit`); the table's TCL-θ row was
         trained this way.
  time   TCL-t (24 equal-time segments), then TCL-θ, both trained quietly.

Budgets follow the reference: FastICA, and TCL's ICA on its learned features,
extract up to ``min(3 * n_gt, 108)`` sources; iVAE has one latent per
ground-truth unit. TCL seeds ``torch`` when it builds its classifier, and iVAE
is initialised from the RNG state TCL leaves, so within a recording the method
order is part of the protocol. Recordings are independent of each other.

Example:
    python scripts/nonlinear_ica.py --data-dir /path/to/benchmark_v2/ch070 \\
        --recordings experiments/lists/nonlinear_ica_ch070.txt \\
        --protocol angle --output results/ch070_nonlinear_ica.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from harmonica.baselines.ivae import run_ivae
from harmonica.baselines.tcl import run_tcl
from harmonica.evaluate import evaluate_decomposition
from harmonica.nonstationary.pipelines import preprocess, run_ica

F1_MIN = 0.5      # a unit counts as recovered at F1 > F1_MIN
R = 4             # Toeplitz extension factor of the shared input
ICA_SEED = 1909
FIELDS = ["dataset", "n_gt", "method", "n_candidates", "n_recovered", "mean_f1", "t_s"]


def load_recording(path: Path):
    """Return ``(emg (C, T) float32, fs, {gt_id: spikes}, angle or None)``."""
    d = np.load(path, allow_pickle=True)
    emg = d["emg"]
    if emg.shape[0] > emg.shape[1]:
        emg = emg.T
    gt = {
        int(k.split("_")[-1]): np.asarray(d[k])
        for k in d.files
        if k.startswith("spike_mu_") and len(d[k]) > 0
    }
    angle = d["angle_profile"].astype(np.float32) if "angle_profile" in d.files else None
    return emg.astype(np.float32), int(d["fs"]), gt, angle


def score(spikes: dict, gt: dict, fs: int) -> tuple[int, float]:
    """``(units recovered at F1 > F1_MIN, mean F1 over those units)``."""
    best: dict[int, float] = {}
    for ps in evaluate_decomposition(spikes, gt, fs)["per_source"]:
        g = ps["gt_id"]
        if g >= 0:
            best[g] = max(best.get(g, 0.0), ps["f1"])
    hits = [f for f in best.values() if f > F1_MIN]
    return len(hits), float(np.mean(hits)) if hits else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="One grid of the benchmark (e.g. benchmark_v2/ch070).")
    ap.add_argument("--recordings", type=Path, required=True,
                    help="Text file with one recording stem per line.")
    ap.add_argument("--protocol", choices=("angle", "time"), default="angle")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tcl-epochs", type=int, default=200)
    ap.add_argument("--ivae-epochs", type=int, default=300)
    ap.add_argument("--n-bins", type=int, default=24,
                    help="Angle bins / time segments of the auxiliary variable.")
    ap.add_argument("--seed", type=int, default=42,
                    help="numpy and torch seed set once at start-up.")
    args = ap.parse_args()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    names = [s.strip() for s in args.recordings.read_text().splitlines() if s.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name in names:
        npz = args.data_dir / f"{name}.npz"
        if not npz.exists():
            print(f"[skip] {name}: missing", flush=True)
            continue
        emg, fs, gt, angle = load_recording(npz)
        if angle is None:
            print(f"[skip] {name}: no angle_profile", flush=True)
            continue
        n_gt = len(gt)
        budget = min(3 * n_gt, 108)
        tcl = dict(n_segments=args.n_bins, n_epochs=args.tcl_epochs, ext_fact=R,
                   sil_th=0.6, ica_seed=ICA_SEED, ica_n_iter=budget)
        print(f"\n=== {name}  n_gt={n_gt}  emg={emg.shape}  protocol={args.protocol}",
              flush=True)

        if args.protocol == "angle":
            runs = [
                ("ICA (linear)", lambda: run_ica(preprocess(emg, R=R)[0], fs, n_gt,
                                                 seed=ICA_SEED)[0]),
                ("TCL-θ", lambda: run_tcl(emg, fs, angle, verbose=True, **tcl)),
                ("iVAE-θ", lambda: run_ivae(emg, fs, angle, n_sources=n_gt,
                                            n_bins=args.n_bins, n_epochs=args.ivae_epochs,
                                            sparsity_weight=10.0, ext_fact=R, verbose=True)),
            ]
        else:
            runs = [
                ("TCL-t", lambda: run_tcl(emg, fs, None, **tcl)),
                ("TCL-θ", lambda: run_tcl(emg, fs, angle, **tcl)),
            ]

        for method, fn in runs:
            t0 = time.time()
            spikes = fn()
            n_rec, mean_f1 = score(spikes, gt, fs)
            row = dict(dataset=name, n_gt=n_gt, method=method, n_candidates=len(spikes),
                       n_recovered=n_rec, mean_f1=round(mean_f1, 4),
                       t_s=round(time.time() - t0, 1))
            rows.append(row)
            print(f"  {method:13s} candidates={len(spikes):3d}  recovered={n_rec:3d}  "
                  f"mean F1={mean_f1:.3f}  ({row['t_s']:.0f} s)", flush=True)

        with args.output.open("w", newline="") as fo:   # rewritten after every recording
            w = csv.DictWriter(fo, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'method':13s} {'recs':>4s} {'cand.':>6s} {'recovered':>9s} {'mean F1':>8s}"
          f"   per recording (F1 > {F1_MIN})")
    for method in dict.fromkeys(r["method"] for r in rows):
        mr = [r for r in rows if r["method"] == method]
        f1s = [r["mean_f1"] for r in mr if r["n_recovered"] > 0]
        print(f"{method:13s} {len(mr):4d} {np.mean([r['n_candidates'] for r in mr]):6.1f} "
              f"{np.mean([r['n_recovered'] for r in mr]):9.1f} "
              f"{np.mean(f1s) if f1s else 0.0:8.3f}   "
              + "/".join(str(r["n_recovered"]) for r in mr))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
