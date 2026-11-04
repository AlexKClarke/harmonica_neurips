#!/usr/bin/env python
"""Per-recording yield tables and significance tests of the supplement.

Per recording and seed, a method's yield is the number of ground-truth units
with best F1 >= the threshold, zero-filled over every recording of the grid.
A recording's yield is the mean over the method's seeds. Methods listed in
``--restrict`` (default: kramberger) are scored only on the recordings their
inputs cover, as in the paper. Pool-size tiers are ``n_gt`` < 40, 40-69, >= 70.

- ``detailed_yield.csv``: mean per-recording yield at each of ``--floors``
  (tab:supp:detailed_yield).
- ``yield_per_rec.csv``: per tier, mean ± s.d. across recordings of the
  per-recording yield, and its median (tab:supp:yield_per_rec).
- ``seed_std.csv``: per tier, the mean yield over the tier's recordings for
  each seed, then mean ± s.d. across seeds (tab:supp:seed_std).
- ``wilcoxon.csv``: ``scipy.stats.wilcoxon`` (default ``zero_method``) on the
  per-recording difference ``--reference`` minus each other method, two-sided
  ``p`` and one-sided ``p_greater``, with the mean difference (Delta) and
  wins / losses / ties (tab:supp:wilcoxon).
- ``bootstrap.csv``: 95 % percentile interval of the mean per-recording
  yield, 5000 resamples, ``numpy.random.default_rng(0)`` (tab:supp:bootstrap).
- ``per_recording_yield.csv``: the per-recording yields themselves.

The means of ``detailed_yield.csv`` and ``yield_per_rec.csv`` are taken in one
step over the (recording, seed) counts: the same value as the mean of the
per-recording yields, rounded once. All but ``detailed_yield.csv`` use
``--threshold``. Methods are given as
``--input METHOD GLOB [GLOB ...]`` (the method named ``ica`` reads the
``f1_ica`` column of a sweep). ``experiments/figures.sh`` has the paper's
inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inputs import parse_inputs, read_units  # noqa: E402

TIERS = {"<40": (0, 40), "40-69": (40, 70), ">=70": (70, np.inf)}


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` at ``1 - alpha`` coverage."""
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(values)), float(lo), float(hi)


def seed_yields(units: pd.DataFrame, recordings: list[str], thr: float) -> pd.DataFrame:
    """Count of units with F1 >= thr per recording (rows) and seed (columns), zero-filled."""
    hit = units.dropna(subset=["f1"])
    hit = hit[hit["f1"] >= thr].drop_duplicates(["dataset", "seed", "gt_mu"])
    seeds = sorted(units["seed"].unique())
    return (hit.groupby(["dataset", "seed"]).size().unstack("seed")
            .reindex(index=recordings, columns=seeds).fillna(0.0))


def wilcoxon(a: pd.Series, b: pd.Series) -> dict:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = both["a"].to_numpy() - both["b"].to_numpy()
    res = sps.wilcoxon(d)
    return {"n_rec": len(d), "delta": float(d.mean()), "p": float(res.pvalue),
            "p_greater": float(sps.wilcoxon(d, alternative="greater").pvalue), "W": float(res.statistic),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", nargs="+", action="append", required=True,
                    metavar=("METHOD", "GLOB"), help="method name and its result CSVs (repeatable)")
    ap.add_argument("--reference", default="harmonica", help="method tested against every other")
    ap.add_argument("--restrict", nargs="*", default=["kramberger"],
                    help="methods compared only on the recordings their inputs cover")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--floors", type=float, nargs="+", default=[0.99, 0.95, 0.90, 0.80],
                    help="F1 floors of detailed_yield.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {m: read_units(globs, ica=(m == "ica")) for m, globs in parse_inputs(args.input).items()}
    if args.reference not in loaded:
        raise SystemExit(f"--reference {args.reference} has no --input")
    grids = sorted({g for _, ev in loaded.values() for g in ev["grid"].unique()})
    n_gt = (pd.concat([u[["grid", "dataset", "n_gt"]] for u, _ in loaded.values()])
            .drop_duplicates(["grid", "dataset"]).set_index(["grid", "dataset"])["n_gt"])
    tables, wil, boot, detailed, per_rec, seed_sd = [], [], [], [], [], []
    for g in grids:
        recs = sorted({d for _, ev in loaded.values() for d in ev.loc[ev.grid == g, "dataset"]})
        tier = pd.Series({d: next(t for t, (lo, hi) in TIERS.items() if lo <= n_gt[(g, d)] < hi)
                          for d in recs})
        Y = {}
        for m, (units, ev) in loaded.items():
            if not (ev.grid == g).any():
                continue
            u = units[units.grid == g]
            covered = pd.Series(recs, index=recs).isin(set(ev.loc[ev.grid == g, "dataset"]))
            keep = covered if m in args.restrict else pd.Series(True, index=recs)
            S = seed_yields(u, recs, args.threshold)[keep]
            Y[m] = S.mean(axis=1).reindex(recs)
            row = {"grid": g, "method": m, "n_rec": int(keep.sum())}
            for f in args.floors:
                row[f"F1>={f:.2f}"] = seed_yields(u, recs, f)[keep].to_numpy().mean()
            detailed.append(row)
            for t in TIERS:
                y, s = Y[m][tier == t].dropna(), S[tier[keep] == t]
                per_rec.append({"grid": g, "method": m, "tier": t, "n_rec": len(y),
                                "mean": s.to_numpy().mean(), "sd": y.std(ddof=1), "median": y.median()})
                seed_means = s.mean(axis=0)
                seed_sd.append({"grid": g, "method": m, "tier": t, "n_seeds": len(seed_means),
                                "mean": seed_means.mean(), "sd": seed_means.std(ddof=1)})
        yt = pd.DataFrame(Y)
        yt.insert(0, "n_gt", [n_gt[(g, d)] for d in yt.index])
        yt.insert(0, "grid", g)
        tables.append(yt)
        for m, s in Y.items():
            v = s.dropna().to_numpy()
            mu, lo, hi = bootstrap_ci(v)
            boot.append({"grid": g, "method": m, "n_rec": len(v), "mean": mu, "ci_lo": lo, "ci_hi": hi})
        for m in Y:
            if m != args.reference:
                wil.append({"grid": g, "reference": args.reference, "baseline": m,
                            **wilcoxon(Y[args.reference], Y[m])})
        print(f"[{g}] {len(recs)} recordings")

    pd.concat(tables).rename_axis("recording").reset_index().to_csv(
        args.out_dir / "per_recording_yield.csv", index=False)
    out = {"detailed_yield": detailed, "yield_per_rec": per_rec, "seed_std": seed_sd,
           "wilcoxon": wil, "bootstrap": boot}
    pd.set_option("display.width", 250, "display.max_rows", 200)
    for name, rows in out.items():
        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"=== {name} ===")
        if name == "wilcoxon":
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        else:
            print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
