#!/usr/bin/env python
"""Single-config hparam sweep — one method + one kwargs set → one CSV.

Runs the specified per-unit refinement method on every recording in
``--val-set`` with custom per-method kwargs, and writes a long-format CSV
with P, R, F1 for the unrefined ICA source (``*_ica`` columns) and for the
refined source. Each ground-truth unit is refined from its best-matching
ICA source (an oracle pairing, used for the per-unit ablations). One
invocation = one config.

Parallelise by launching many invocations (e.g., SLURM array), one per
config.

Example:
    python scripts/bench_sweep.py \\
        --data-dir /path/ch070 \\
        --val-set experiments/lists/ch070_all.txt \\
        --method iterative_refractory \\
        --kwargs '{"mc":0.3,"lam":2.0}' \\
        --tag iref_mc03_lam2 \\
        --output results/sweep_phase1/iref_mc03_lam2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from harmonica.evaluate import evaluate_decomposition
from harmonica.models.harmonica_signal_space import (
    refine_signal_space_comp_warmstart,
)
from harmonica.ica.algorithm import peel_off as _peel_off_classical
from harmonica.nonstationary.detect import detect_best
from harmonica.nonstationary.harmonica import (
    iternn_refine,
    persample_refine,
)
from harmonica.nonstationary.pipelines import _prf, preprocess, run_ica
from harmonica.nonstationary.refractory import (
    refine_harmonica_style,
    refine_iterative_refractory,
)

# Methods that take anchor_spikes (5th positional arg)
ANCHORED = {
    "iterative_refractory": refine_iterative_refractory,
    "harmonica_style":      refine_harmonica_style,
    "signal_space":         refine_signal_space_comp_warmstart,
}
# Methods that do not take anchors
ANCHORLESS = {
    "timenn": persample_refine,   # kwargs: epochs, loss_type, max_corr
    "iternn": iternn_refine,      # kwargs: mc_schedule, ep_schedule, select, loss_type, fsamp
}


def load_recording(path: Path) -> tuple[np.ndarray, int, dict]:
    d = np.load(path, allow_pickle=True)
    fs_key = "fs" if "fs" in d.files else "FS"
    FS = int(d[fs_key])
    emg = d["emg"]
    if emg.shape[0] > emg.shape[1]:
        emg = emg.T
    emg = emg.astype(np.float32)
    prefix = "spike_mu_" if any(k.startswith("spike_mu_") for k in d.files) else "gt_spikes_"
    gt = {
        int(k.split("_")[-1]): np.asarray(d[k])
        for k in d.files
        if k.startswith(prefix) and len(d[k]) > 0
    }
    return emg, FS, gt


def run_method(method: str, kwargs: dict, z_w, w_ica, t_norm, FS, anchor_sp, nn_seed):
    if method in ANCHORED:
        fn = ANCHORED[method]
        return fn(z_w, w_ica, t_norm, FS, anchor_sp, seed=nn_seed, **kwargs)
    if method == "timenn":
        return persample_refine(z_w, w_ica, t_norm, seed=nn_seed, **kwargs)
    if method == "iternn":
        # iternn_refine returns (filters, spikes_dict, source, sils)
        _, _, source, _ = iternn_refine(
            z_w, w_ica, t_norm, fsamp=FS, seed=nn_seed, **kwargs
        )
        return source
    raise ValueError(f"unknown method: {method}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--val-set",  type=Path, required=True,
                    help="Text file with one dataset stem per line")
    ap.add_argument("--method", required=True,
                    choices=sorted(list(ANCHORED) + list(ANCHORLESS)))
    ap.add_argument("--kwargs", default="{}",
                    help="JSON dict of method kwargs (e.g. '{\"mc\":0.3}')")
    ap.add_argument("--tag", required=True,
                    help="Config tag written to every row")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ica-seed", type=int, default=1909)
    ap.add_argument("--nn-seed",  type=int, default=42)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--d-eff", type=int, default=128)
    ap.add_argument(
        "--peel-mode",
        choices=("none", "ica", "refine"),
        default="none",
        help=(
            "Default-style (classical STA) peel-off ablation: "
            "'none' = no peel-off (historical default); "
            "'ica' = enable peel-off inside the ICA loop only; "
            "'refine' = peel each MU's STA from z_w after refinement, "
            "next MU is refined on the residual (filters processed in "
            "ICA-discovery order)."
        ),
    )
    args = ap.parse_args()

    kwargs = json.loads(args.kwargs)
    val_names = [l.strip() for l in args.val_set.read_text().splitlines() if l.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] tag={args.tag}  method={args.method}  kwargs={kwargs}")
    print(f"[sweep] val_set={len(val_names)} datasets  ica_seed={args.ica_seed} nn_seed={args.nn_seed}")
    print(f"[sweep] peel_mode={args.peel_mode}  output={args.output}")
    sys.stdout.flush()

    fieldnames = [
        "tag", "method", "dataset", "gt_mu", "eid", "n_gt",
        "p_ica", "r_ica", "f1_ica",
        "p",    "r",    "f1",
        "t_dataset_s",
    ]

    t_all = time.time()
    with args.output.open("w", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for idx, name in enumerate(val_names, 1):
            npz = args.data_dir / f"{name}.npz"
            if not npz.exists():
                print(f"[{idx}/{len(val_names)}] {name}  MISSING")
                sys.stdout.flush()
                continue
            try:
                emg, FS, gt = load_recording(npz)
                T = emg.shape[1]
                t0 = time.time()
                z_w, _ = preprocess(emg, R=args.R, d_eff=args.d_eff)
                spk, _sils, filters, t_norm = run_ica(
                    z_w, FS, len(gt),
                    seed=args.ica_seed,
                    peel_off=(args.peel_mode == "ica"),
                )
                ev = evaluate_decomposition(spk, gt, FS)
                best: dict = {}
                for ps in ev["per_source"]:
                    gid = ps["gt_id"]
                    if gid >= 0 and (gid not in best or ps["f1"] > best[gid]["f1"]):
                        best[gid] = ps

                # In refine-mode we peel each MU off z_w after refinement, so
                # later filters are refined on a residual signal. Process in
                # ICA-discovery order (ascending est_id) so the peel sequence
                # mirrors the ICA discovery sequence.
                items = sorted(
                    best.items(), key=lambda kv: int(kv[1]["est_id"])
                )
                z_w_local = z_w if args.peel_mode != "refine" else z_w.copy()

                n_rows = 0
                for gid, ps in items:
                    eid = ps["est_id"]
                    w_ica = filters[:, eid] / np.linalg.norm(filters[:, eid])
                    ica_sp = spk[eid].astype(int)
                    source = run_method(
                        args.method, kwargs,
                        z_w_local, w_ica, t_norm, FS, ica_sp, args.nn_seed,
                    )
                    sp, _ = detect_best(source, FS, T)
                    p, r, f1 = _prf(sp, gt[gid], FS)
                    if args.peel_mode == "refine" and len(sp) > 0:
                        z_w_local, _, _ = _peel_off_classical(
                            z_w_local, sp.astype(int), fsamp=float(FS),
                        )
                    writer.writerow({
                        "tag": args.tag, "method": args.method,
                        "dataset": name, "gt_mu": gid, "eid": eid,
                        "n_gt": len(gt),
                        "p_ica": f"{ps['precision']:.6f}",
                        "r_ica": f"{ps['recall']:.6f}",
                        "f1_ica": f"{ps['f1']:.6f}",
                        "p": f"{p:.6f}", "r": f"{r:.6f}", "f1": f"{f1:.6f}",
                        "t_dataset_s": f"{time.time() - t0:.2f}",
                    })
                    n_rows += 1
                fo.flush()
                print(
                    f"[{idx}/{len(val_names)}] {name}  "
                    f"gt={len(gt):3d} matched={len(best):3d}  "
                    f"{time.time() - t0:6.1f}s  +{n_rows} rows"
                )
                sys.stdout.flush()
            except Exception as e:
                print(f"[{idx}/{len(val_names)}] {name}  ERROR: {type(e).__name__}: {e}")
                sys.stdout.flush()

    print(f"[sweep] done in {time.time() - t_all:.1f}s → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
