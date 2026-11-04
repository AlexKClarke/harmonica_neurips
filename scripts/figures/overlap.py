#!/usr/bin/env python
"""Which motor units do the methods share? Supplementary overlap figures.

A method *accepts* a ground-truth unit on a (grid, seed, recording) if its
best-matching source reaches F1 >= 0.95. Seeds are paired by index (seed k of
every runner uses FastICA seed 1909 + k); a method without a given seed has an
empty set there.

- ``heatmap``: for every method pair (A, B), pooled over all (grid, seed,
  recording) keys, Jaccard = sum|A n B| / sum|A u B| and the share of A's
  units also accepted by B = sum|A n B| / sum|A|.
  Writes ``pairwise_overlap_heatmap.{pdf,png}``,
  ``pairwise_overlap_heatmap_by_density.png``, ``pairwise_overlap_summary.csv``
  and ``pairwise_overlap_by_density.csv``.
- ``bars``: per method, grid and pool-size tier, the mean number of accepted
  units per recording-seed that ICA also accepts, that only the method
  accepts, and that only ICA accepts.
  Writes ``mu_overlap_stacked_bars.{pdf,png}``, ``mu_overlap_by_bin.csv`` and
  ``mu_overlap_by_method.csv``.

Methods are given as ``--input METHOD GLOB [GLOB ...]`` with the names
harmonica, ica (``f1_ica`` column of a sweep), chen, kramberger, yeung,
glaser, adapt. ``experiments/figures.sh`` has the paper's inputs.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inputs import parse_inputs, read_units  # noqa: E402

METHOD_ORDER = ["harmonica", "ica", "chen", "kramberger", "yeung", "glaser", "adapt"]
BAR_METHODS = ["harmonica", "chen", "yeung", "kramberger", "glaser", "adapt"]
BINS = {"<40": (0, 40), "40-69": (40, 70), ">=70": (70, 9999)}

SetDict = dict  # (grid, seed, recording) -> set of accepted GT ids


def accepted_sets(units: pd.DataFrame, thr: float) -> SetDict:
    """``{(grid, seed, recording): accepted GT ids}``, only for non-empty sets."""
    hit = units.dropna(subset=["f1"])
    hit = hit[hit["f1"] >= thr]
    return {(g, int(s), d): set(int(x) for x in sub["gt_mu"])
            for (g, s, d), sub in hit.groupby(["grid", "seed", "dataset"])}


def pairwise(sources: dict[str, SetDict], order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled (all keys) and per-grid overlap of every method pair.

    Keys are the (grid, seed, recording) triples on which at least one method
    accepts a unit.
    """
    by_rec: dict[tuple, dict[str, set[int]]] = {}
    for m, sets in sources.items():
        for k, s in sets.items():
            by_rec.setdefault(k, {})[m] = s
    rows = []
    for (grid, seed, ds), per in by_rec.items():
        for a, b in product(order, order):
            sa, sb = per.get(a, set()), per.get(b, set())
            rows.append({"density": grid, "seed": seed, "dataset": ds, "method_A": a, "method_B": b,
                         "n_A": len(sa), "n_B": len(sb), "n_intersect": len(sa & sb),
                         "n_union": len(sa | sb)})
    long = pd.DataFrame(rows)
    out = []
    for keys in (["method_A", "method_B"], ["density", "method_A", "method_B"]):
        cols = ["n_A", "n_B", "n_intersect", "n_union"]
        t = long.groupby(keys)[cols].mean().round(3).reset_index()
        s = long.groupby(keys)[cols].sum().reset_index()
        t["jaccard"] = s["n_intersect"] / s["n_union"].clip(lower=1e-9)
        t["pct_A_in_B"] = 100 * s["n_intersect"] / s["n_A"].clip(lower=1e-9)
        t["n_keys"] = long.groupby(keys).size().values
        out.append(t)
    return out[0], out[1]


def matrix(summ: pd.DataFrame, value: str, order: list[str]) -> np.ndarray:
    s = summ.set_index(["method_A", "method_B"])
    mat = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if (a, b) in s.index:
                mat[i, j] = s.loc[(a, b), value]
    return mat


def assign_bin(n_gt: int) -> str:
    for label, (lo, hi) in BINS.items():
        if lo <= n_gt < hi:
            return label
    return "??"


def with_ica(sources: dict[str, SetDict], n_gt: dict, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per method: units shared with ICA, method-only and ICA-only, mean per recording-seed."""
    ica = sources["ica"]
    grids = sorted({k[0] for k in ica})
    seeds = sorted({k[1] for s in sources.values() for k in s})
    rows = []
    for m in methods:
        sets = sources[m]
        for g in grids:
            for seed in seeds:
                ms = {k[2]: v for k, v in sets.items() if k[0] == g and k[1] == seed}
                if not ms:
                    continue
                isets = {k[2]: v for k, v in ica.items() if k[0] == g and k[1] == seed}
                for ds in set(ms) | set(isets):
                    s_m, s_i = ms.get(ds, set()), isets.get(ds, set())
                    rows.append({"method": m, "density": g, "seed": seed, "dataset": ds,
                                 "bin": assign_bin(n_gt[(g, ds)]),
                                 "n_method": len(s_m), "n_ica": len(s_i), "n_both": len(s_m & s_i),
                                 "n_method_only": len(s_m - s_i), "n_ica_only": len(s_i - s_m)})
    long = pd.DataFrame(rows)
    cols = ["n_method", "n_ica", "n_both", "n_method_only", "n_ica_only"]
    by_bin = long.groupby(["method", "density", "bin"])[cols].mean().round(3).reset_index()
    by_method = long.groupby(["method", "density"])[cols].mean().round(3).reset_index()
    for t in (by_bin, by_method):
        t["pct_method_in_ica"] = (100 * t["n_both"] / t["n_method"].clip(lower=1e-9)).round(1)
        t["pct_method_novel"] = (100 * t["n_method_only"] / t["n_method"].clip(lower=1e-9)).round(1)
    return by_bin, by_method


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def annotate(ax, mat, fmt="{:.2f}"):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black")


def _heat(ax, M, order, cmap, vmax, fmt, title, fig):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    annotate(ax, M, fmt=fmt)
    fig.colorbar(im, ax=ax, shrink=0.8)


def plot_heatmap(summ: pd.DataFrame, order: list[str], grids: list[str], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _heat(axes[0], matrix(summ, "jaccard", order), order, "Blues", 1, "{:.2f}",
          "Jaccard similarity\n(|A ∩ B| / |A ∪ B|)", fig)
    _heat(axes[1], matrix(summ, "pct_A_in_B", order), order, "Greens", 100, "{:.0f}",
          "Asymmetric overlap\n(% of row method's MUs also in column)", fig)
    fig.suptitle("Pairwise MU-set overlap at F1 ≥ 0.95 (averaged over recordings × seeds × "
                 f"{'+'.join(grids)})")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=140)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[fig] {stem}.{{pdf,png}}")


def plot_heatmap_by_density(per_dens: pd.DataFrame, order: list[str], grids: list[str], stem: Path) -> None:
    fig, axes = plt.subplots(2, len(grids), figsize=(13, 11), squeeze=False)
    for col, g in enumerate(grids):
        sub = per_dens[per_dens["density"] == g]
        _heat(axes[0, col], matrix(sub, "jaccard", order), order, "Blues", 1, "{:.2f}",
              f"Jaccard — {g}", fig)
        _heat(axes[1, col], matrix(sub, "pct_A_in_B", order), order, "Greens", 100, "{:.0f}",
              f"% of row's MUs in column — {g}", fig)
    fig.suptitle("Pairwise MU-set overlap at F1 ≥ 0.95, stratified by electrode density")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"[fig] {stem}.png")


def plot_bars(by_bin: pd.DataFrame, methods: list[str], grids: list[str], thr: float, stem: Path) -> None:
    fig, axes = plt.subplots(1, len(grids), figsize=(12, 5), sharey=True, squeeze=False)
    axes = axes[0]
    bins_order = list(BINS)
    for ax, g in zip(axes, grids):
        sub = by_bin[by_bin["density"] == g].set_index(["method", "bin"])
        x = np.arange(len(methods))
        width = 0.25
        for i, b in enumerate(bins_order):
            both, novel, miss = [], [], []
            for m in methods:
                if (m, b) in sub.index:
                    both.append(sub.loc[(m, b), "n_both"])
                    novel.append(sub.loc[(m, b), "n_method_only"])
                    miss.append(sub.loc[(m, b), "n_ica_only"])
                else:
                    both.append(0); novel.append(0); miss.append(0)
            offset = (i - 1) * width
            ax.bar(x + offset, both, width, color="#1f77b4", label="both find" if i == 0 else None)
            ax.bar(x + offset, novel, width, bottom=both, color="#2ca02c",
                   label=r"method only ($\Delta$ vs ICA)" if i == 0 else None)
            ax.bar(x + offset, miss, width, bottom=np.array(both) + np.array(novel),
                   color="#d62728", alpha=0.6, label="ICA only (method misses)" if i == 0 else None)
            for k, val in enumerate(np.array(both) + np.array(novel)):
                ax.text(x[k] + offset, val + 0.05, b, ha="center", fontsize=6, rotation=90, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(f"{g} — per-recording mean MUs at F1 >= {thr}")
    axes[0].set_ylabel(f"mean MUs / recording (yield@F1>={thr})")
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
    fig.suptitle("Stacked: where do each method's MUs sit relative to ICA's accepted set?")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(stem.with_suffix(".png"), dpi=140)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[fig] {stem}.{{pdf,png}}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figure", choices=["heatmap", "bars", "both"])
    ap.add_argument("--input", nargs="+", action="append", required=True,
                    metavar=("METHOD", "GLOB"), help="method name and its result CSVs (repeatable)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_inputs(args.input)
    unknown = set(inputs) - set(METHOD_ORDER)
    if unknown:
        raise SystemExit(f"unknown method(s) {sorted(unknown)}; choose from {METHOD_ORDER}")
    sources, n_gt = {}, {}
    for m, globs in inputs.items():
        units, _ = read_units(globs, ica=(m == "ica"))
        sources[m] = accepted_sets(units, args.threshold)
        for (g, d), n in units.groupby(["grid", "dataset"])["n_gt"].first().items():
            n_gt.setdefault((g, d), int(n))
    order = [m for m in METHOD_ORDER if m in sources]
    grids = sorted({k[0] for s in sources.values() for k in s})
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)

    if args.figure in ("heatmap", "both"):
        summ, per_dens = pairwise(sources, order)
        summ.to_csv(args.out_dir / "pairwise_overlap_summary.csv", index=False)
        per_dens.to_csv(args.out_dir / "pairwise_overlap_by_density.csv", index=False)
        plot_heatmap(summ, order, grids, args.out_dir / "pairwise_overlap_heatmap")
        plot_heatmap_by_density(per_dens, order, grids, args.out_dir / "pairwise_overlap_heatmap_by_density")
        for value, digits in (("jaccard", 3), ("pct_A_in_B", 1)):
            print(f"=== {value} (row A, column B) ===")
            print(pd.DataFrame(matrix(summ, value, order), index=order, columns=order)
                  .round(digits).to_string())
    if args.figure in ("bars", "both"):
        if "ica" not in sources:
            raise SystemExit("bars needs an ica input")
        methods = [m for m in BAR_METHODS if m in sources]
        by_bin, by_method = with_ica(sources, n_gt, methods)
        by_bin.to_csv(args.out_dir / "mu_overlap_by_bin.csv", index=False)
        by_method.to_csv(args.out_dir / "mu_overlap_by_method.csv", index=False)
        plot_bars(by_bin, methods, grids, args.threshold, args.out_dir / "mu_overlap_stacked_bars")
        print(by_method.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
