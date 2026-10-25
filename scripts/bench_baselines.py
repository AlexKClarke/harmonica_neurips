#!/usr/bin/env python
"""Single-config baseline sweep — one baseline method + kwargs → one CSV.

Baselines (:mod:`harmonica.baselines`) are self-contained: each
``run_<method>(emg, FS, **kwargs)`` returns ``{mu_id: spike_indices}``
with internal calibration + adaptation. We score the returned dict
against the recording's GT via :func:`evaluate_decomposition` and record
per-GT-MU best P/R/F1.

Example:
    python scripts/bench_baselines.py \\
        --data-dir /path/ch070 \\
        --val-set  experiments/lists/ch070_all.txt \\
        --method   glaser \\
        --kwargs   '{"window_s":2.0}' \\
        --tag      glaser_default \\
        --output   results/baselines_val/glaser_default.csv
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

from harmonica.baselines import (
    run_adapt,
    run_chen,
    run_glaser,
    run_glaser_cyclostationary,
    run_ivae,
    run_kramberger,
    run_signal_space,
    run_tcl,
    run_yeung,
)
from harmonica.evaluate import evaluate_decomposition

METHOD_FNS = {
    "glaser":              run_glaser,
    "glaser_cyclostationary": run_glaser_cyclostationary,  # paper-faithful Pseudocode 1
    "kramberger":          run_kramberger,   # angle required
    "chen":                run_chen,
    "yeung":               run_yeung,
    "adapt":               run_adapt,
    "signal_space":        run_signal_space,        # HarmonICA: multi-MU + neural peel-off, spike/null contrast
    "tcl":                 run_tcl,                 # Hyvarinen-Morioka time-contrastive nonlinear ICA (angle optional)
    "ivae":                run_ivae,                # Khemakhem identifiable VAE nonlinear ICA (u=angle, optional)
}
# Methods that REQUIRE angle (skip dataset if missing).
NEEDS_ANGLE = {"kramberger"}
# Methods that ACCEPT angle as an optional 3rd positional arg (used when
# the npz has it; falls back to internal default — sinusoidal positional
# encoding for signal_space; angle-bin vs equal-time segments
# for tcl / ivae).
OPTIONAL_ANGLE = {"signal_space", "tcl", "ivae"}


def load_recording(path: Path) -> tuple[np.ndarray, int, dict, np.ndarray | None]:
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
    angle = None
    for key in ("angle_profile", "angle_continuous"):
        if key in d.files:
            angle = np.asarray(d[key], dtype=np.float32)
            break
    return emg, FS, gt, angle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--val-set",  type=Path, required=True)
    ap.add_argument("--method",   required=True, choices=sorted(METHOD_FNS.keys()))
    ap.add_argument("--kwargs",   default="{}",
                    help="JSON dict of method kwargs")
    ap.add_argument("--tag",      required=True)
    ap.add_argument("--output",   type=Path, required=True)
    ap.add_argument(
        "--angle-mode",
        choices=("real", "shuffled", "constant", "none"),
        default="real",
        help=(
            "Context-variable ablation for OPTIONAL_ANGLE methods. "
            "'real' = use angle from npz (default); "
            "'shuffled' = randomly permute angle in time; "
            "'constant' = replace angle with zeros (static A(t)); "
            "'none' = drop angle and let the method fall back to "
            "its internal sinusoidal positional encoding."
        ),
    )
    ap.add_argument("--angle-mode-seed", type=int, default=2026,
                    help="RNG seed for --angle-mode shuffled.")
    args = ap.parse_args()

    kwargs = json.loads(args.kwargs)
    val_names = [l.strip() for l in args.val_set.read_text().splitlines() if l.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] tag={args.tag}  method={args.method}  kwargs={kwargs}")
    print(f"[baseline] val_set={len(val_names)}  output={args.output}")
    print(f"[baseline] angle_mode={args.angle_mode}  angle_mode_seed={args.angle_mode_seed}")
    sys.stdout.flush()
    rng = np.random.default_rng(args.angle_mode_seed)

    fieldnames = [
        "tag", "method", "dataset", "gt_mu", "eid", "n_gt",
        "p", "r", "f1", "t_dataset_s",
    ]
    fn = METHOD_FNS[args.method]
    needs_angle = args.method in NEEDS_ANGLE
    optional_angle = args.method in OPTIONAL_ANGLE

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
                emg, FS, gt, angle = load_recording(npz)
                if needs_angle and angle is None:
                    print(f"[{idx}/{len(val_names)}] {name}  SKIP (no angle)")
                    sys.stdout.flush()
                    continue
                # Apply --angle-mode for OPTIONAL_ANGLE methods (context ablation).
                if optional_angle and angle is not None:
                    if args.angle_mode == "shuffled":
                        perm = rng.permutation(len(angle))
                        angle = angle[perm].astype(np.float32)
                    elif args.angle_mode == "constant":
                        angle = np.zeros_like(angle, dtype=np.float32)
                    elif args.angle_mode == "none":
                        angle = None
                t0 = time.time()
                kw_call = dict(kwargs)
                if needs_angle:
                    spikes = fn(emg, FS, angle, **kw_call)
                elif optional_angle:
                    # Pass angle when the npz has it; let the method
                    # fall back to its own default otherwise.
                    spikes = fn(emg, FS, angle, **kw_call)
                else:
                    spikes = fn(emg, FS, **kw_call)
                ev = evaluate_decomposition(spikes, gt, FS)
                best: dict = {}
                for ps in ev["per_source"]:
                    gid = ps["gt_id"]
                    if gid >= 0 and (gid not in best or ps["f1"] > best[gid]["f1"]):
                        best[gid] = ps

                n_rows = 0
                for gid, ps in best.items():
                    writer.writerow({
                        "tag": args.tag, "method": args.method,
                        "dataset": name, "gt_mu": gid, "eid": ps["est_id"],
                        "n_gt": len(gt),
                        "p":  f"{ps['precision']:.6f}",
                        "r":  f"{ps['recall']:.6f}",
                        "f1": f"{ps['f1']:.6f}",
                        "t_dataset_s": f"{time.time() - t0:.2f}",
                    })
                    n_rows += 1
                fo.flush()
                print(
                    f"[{idx}/{len(val_names)}] {name}  "
                    f"gt={len(gt):3d} method_mus={len(spikes):3d} matched={len(best):3d}  "
                    f"{time.time() - t0:6.1f}s  +{n_rows} rows"
                )
                sys.stdout.flush()
            except Exception as e:
                import traceback
                print(f"[{idx}/{len(val_names)}] {name}  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                sys.stdout.flush()

    print(f"[baseline] done in {time.time() - t_all:.1f}s → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
