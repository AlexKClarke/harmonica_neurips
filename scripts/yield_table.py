#!/usr/bin/env python
"""Motor-unit yield at F1 >= 0.95, binned by pool size — the paper's tables.

Each input CSV is one seed (the output of ``bench_baselines.py`` or
``bench_sweep.py``). For every file we count the ground-truth units whose
best-match F1 reaches the threshold, split by the recording's ground-truth
pool size ``n_gt`` into ``< 40``, ``40-69`` and ``>= 70``, then report
mean ± s.d. across files. ``--top-k`` adds the mean F1 of the ``k`` best
units per file (the paper's "top-1000 F1").

Example:
    python scripts/yield_table.py --name "HarmonICA (PE)" --top-k 1000 \\
        results/ch320_harmonica_pe_seed*.csv
    python scripts/yield_table.py --name "ICA / cBSS" --column f1_ica \\
        results/ch320_ica_reference_seed*.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BIN_EDGES = [0, 40, 70, np.inf]
BIN_LABELS = ["n_gt<40", "40-69", ">=70"]


def _fmt(values: list[float], digits: int = 1) -> str:
    if len(values) < 2:
        return f"{values[0]:.{digits}f} ± —"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csvs", nargs="+", type=Path, help="one CSV per seed")
    ap.add_argument("--name", default="method")
    ap.add_argument("--column", default="f1",
                    help="F1 column to score ('f1_ica' for the ICA reference row)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0,
                    help="also report mean F1 of the k best units per file")
    args = ap.parse_args()

    per_bin = {b: [] for b in BIN_LABELS}
    totals, topk = [], []
    for path in args.csvs:
        df = pd.read_csv(path).dropna(subset=[args.column])
        hit = df[df[args.column] >= args.threshold]
        bins = pd.cut(hit["n_gt"], BIN_EDGES, labels=BIN_LABELS, right=False)
        counts = bins.value_counts()
        for b in BIN_LABELS:
            per_bin[b].append(float(counts.get(b, 0)))
        totals.append(float(len(hit)))
        if args.top_k:
            best = np.sort(df[args.column].to_numpy())[::-1][: args.top_k]
            topk.append(float(best.mean()))

    cells = [f"{b}: {_fmt(per_bin[b])}" for b in BIN_LABELS]
    line = f"{args.name}  ({len(args.csvs)} files)  " + "  ".join(cells)
    line += f"  total: {_fmt(totals)}"
    if args.top_k:
        line += f"  top-{args.top_k} F1: {_fmt(topk, 3)}"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
