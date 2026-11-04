#!/usr/bin/env python
"""Motor-unit yield binned by pool size, and the other per-seed table columns.

Each input CSV is one seed (the output of ``bench_baselines.py`` or
``bench_sweep.py``), or a long-format ``bench_adapt.py`` CSV, which holds
every seed (``seed_idx``) and is scored on ``best_f1`` over its ``ok`` rows.
For every seed we count the ground-truth units whose best-match F1 reaches
``--threshold``, split by the recording's ground-truth pool size ``n_gt`` into
``< 40``, ``40-69`` and ``>= 70``, then report mean ± s.d. across seeds, and
the total. Optional per-seed columns, also mean ± s.d. across seeds:

- ``--top-k K``: mean F1 of the ``K`` best units (the paper's "top-1000 F1");
- ``--mean-f1``: mean F1 over every unit in the CSV (per-seed CSVs hold the
  matched units, long-format CSVs every unit, unmatched ones at F1 = 0);
- ``--also-at T1,T2,...``: total yield at further F1 thresholds.

Two other modes:

- ``--strata``: recordings, ground-truth units, mean and range of ``n_gt``
  per tier (main Table 1). Pass CSVs that cover every recording, e.g. the
  HarmonICA or ICA-reference CSVs of one seed, or a ``bench_adapt.py`` CSV.
- ``--runtime``: wall time per (recording, seed) run (``t_dataset_s``, or
  ``time_s`` in long format): mean, median, quartiles, 5-95 % range and
  full range over the runs with output.

Example:
    python scripts/yield_table.py --name "HarmonICA (PE)" --top-k 1000 \\
        results/ch320_harmonica_pe_seed*.csv
    python scripts/yield_table.py --name "ICA / cBSS" --column f1_ica \\
        results/ch320_ica_reference_seed*.csv
    python scripts/yield_table.py --name "Mendez Guerra 2024" \\
        results/ch320_adapt_tuned.csv
    python scripts/yield_table.py --name "rank 5" --mean-f1 --also-at 0.85,0.90 \\
        results/ch320_rank5_seed*.csv
    python scripts/yield_table.py --strata --name ch320 results/ch320_harmonica_pe_seed0.csv
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


def _seeds(path: Path, column: str, method: str | None):
    """Yield ``(frame, f1_column)`` once per seed held in ``path``.

    Frames carry ``dataset``, ``n_gt`` and ``t_dataset_s`` in both layouts.
    """
    df = pd.read_csv(path)
    if not {"seed_idx", "best_f1"} <= set(df.columns):
        yield df, column
        return
    # bench_adapt.py long format: one row per GT unit, every seed in one file.
    if method is not None:
        df = df[df["method"] == method]
    elif df["method"].nunique() > 1:
        raise SystemExit(f"{path} holds several methods {sorted(df['method'].unique())}; "
                         "pick one with --method")
    df = df[df["ok"].astype(bool)].rename(columns={"recording": "dataset", "time_s": "t_dataset_s"})
    col = "best_f1" if column == "f1" else column
    for _, g in df.groupby("seed_idx"):
        yield g, col


def summarise(paths: list[Path], column: str = "f1", method: str | None = None,
              threshold: float = 0.95, top_k: int = 0, also_at: tuple[float, ...] = ()) -> dict:
    """Per-seed values of every table column.

    Returns:
        ``{key: [value per seed]}`` with keys ``BIN_LABELS``, ``"total"``,
        ``"mean_f1"``, ``"top_k"`` (if ``top_k``) and ``"n>=T"`` for each
        threshold in ``also_at``.
    """
    out: dict[str, list[float]] = {k: [] for k in [*BIN_LABELS, "total", "mean_f1"]}
    for path in paths:
        for df, col in _seeds(path, column, method):
            df = df.dropna(subset=[col])
            hit = df[df[col] >= threshold]
            counts = pd.cut(hit["n_gt"], BIN_EDGES, labels=BIN_LABELS, right=False).value_counts()
            for b in BIN_LABELS:
                out[b].append(float(counts.get(b, 0)))
            out["total"].append(float(len(hit)))
            out["mean_f1"].append(float(df[col].mean()))
            if top_k:
                out.setdefault("top_k", []).append(float(np.sort(df[col].to_numpy())[::-1][:top_k].mean()))
            for t in also_at:
                out.setdefault(f"n>={t:.2f}", []).append(float((df[col] >= t).sum()))
    return out


def strata(paths: list[Path], method: str | None = None) -> pd.DataFrame:
    """Recordings, GT units, mean and range of ``n_gt`` per tier and in total."""
    rec = pd.concat([df[["dataset", "n_gt"]] for p in paths for df, _ in _seeds(p, "f1", method)])
    rec = rec.drop_duplicates()
    if rec["dataset"].duplicated().any():
        raise SystemExit("inconsistent n_gt for a recording across the input CSVs")
    tiers = pd.cut(rec["n_gt"], BIN_EDGES, labels=BIN_LABELS, right=False)
    cols = {}
    for label, n in [*((b, rec["n_gt"][tiers == b]) for b in BIN_LABELS), ("total", rec["n_gt"])]:
        cols[label] = {"recordings": len(n), "GT units": int(n.sum()),
                       "mean n_gt": f"{n.mean():.1f}", "n_gt range": f"{n.min()}-{n.max()}"}
    return pd.DataFrame(cols)


def runtime(paths: list[Path], method: str | None = None) -> pd.Series:
    """Wall time of each (recording, seed) run: the largest ``t_dataset_s`` of its rows."""
    t = [df.groupby("dataset")["t_dataset_s"].max() for p in paths for df, _ in _seeds(p, "f1", method)]
    return pd.concat(t, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csvs", nargs="+", type=Path,
                    help="one CSV per seed, or bench_adapt.py long-format CSVs")
    ap.add_argument("--name", default="method")
    ap.add_argument("--column", default="f1",
                    help="F1 column to score ('f1_ica' for the ICA reference row)")
    ap.add_argument("--method", default=None,
                    help="method to score in a long-format CSV that holds several")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0,
                    help="also report mean F1 of the k best units per seed")
    ap.add_argument("--mean-f1", action="store_true",
                    help="also report mean F1 over every unit per seed")
    ap.add_argument("--also-at", type=lambda v: [float(t) for t in v.split(",")], default=[],
                    metavar="T1,T2,...", help="also report the total yield at these F1 thresholds")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--strata", action="store_true",
                      help="print recordings, GT units and n_gt per tier instead")
    mode.add_argument("--runtime", action="store_true",
                      help="print the wall time per (recording, seed) run instead")
    args = ap.parse_args()

    if args.strata:
        print(f"{args.name}\n" + strata(args.csvs, args.method).to_string())
        return 0
    if args.runtime:
        t = runtime(args.csvs, args.method)
        q = t.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{args.name}  ({len(t)} runs)  mean {t.mean():.1f} s  median {q[0.5]:.1f} s  "
              f"IQR {q[0.25]:.1f}-{q[0.75]:.1f} s  5-95 %: {q[0.05]:.1f}-{q[0.95]:.1f} s  "
              f"range {t.min():.1f}-{t.max():.1f} s")
        return 0

    s = summarise(args.csvs, args.column, args.method, args.threshold, args.top_k, tuple(args.also_at))
    line = f"{args.name}  ({len(s['total'])} seeds)  " + "  ".join(f"{b}: {_fmt(s[b])}" for b in BIN_LABELS)
    line += f"  total: {_fmt(s['total'])}"
    if args.top_k:
        line += f"  top-{args.top_k} F1: {_fmt(s['top_k'], 3)}"
    if args.mean_f1:
        line += f"  mean F1: {_fmt(s['mean_f1'], 3)}"
    for t in args.also_at:
        line += f"  n>={t:.2f}: {_fmt(s[f'n>={t:.2f}'])}"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
