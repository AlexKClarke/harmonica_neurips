#!/usr/bin/env python
"""Yield x recall (main Fig. 3) and yield x F1 (supplement) Pareto figures.

For each method and grid, every ground-truth unit the method matched at all
(F1 > 0) contributes one value per (recording, seed): the recall of its
best-F1 source (``--axis recall``) or that F1 (``--axis f1``). Duplicate rows
of a unit keep the best-F1 row. The values are pooled, sorted in decreasing
order, and the k-th is drawn at x = k / N, where N is the number of
(recording, seed) pairs the method has rows for (``--denominator matched``,
the paper's figures) or recordings x seeds (``all``). The x where a curve
crosses a threshold T is then the mean number of units per recording-seed at
recall (or F1) >= T; the dot at the end of a curve is the method's coverage.

Methods are given as ``--input METHOD GLOB [GLOB ...]``; the grid of a file
comes from its name (per-seed CSVs) or its ``grid`` column (``bench_adapt.py``).
Drawn methods: harmonica, ica (``f1_ica`` / ``r_ica`` columns of a sweep),
chen, yeung, kramberger, glaser, adapt. Any other method name is tabulated
but not drawn: ``harmonica_theta`` (HarmonICA (θ)) and ``ica_harness`` (the
ICA / cBSS reference scored inside ``bench_adapt.py``).

Writes, per grid and for both grids side by side: ``pareto_<axis>_<grid>.{pdf,png}``,
and ``pareto_<axis>_crossovers.csv`` (units per recording-seed at
T in {0.99, 0.95, 0.90, 0.80}), ``pareto_<axis>_sweep.csv`` and
``pareto_<axis>_dominance.txt`` (the smallest margin of each HarmonICA
variant, harmonica and harmonica_theta, over every drawn baseline at every
threshold, and where Adapt sits relative to ICA and the other baselines).
``experiments/figures.sh`` has the paper's inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inputs import parse_inputs, read_units  # noqa: E402

ORDER = ["harmonica", "ica", "chen", "yeung", "kramberger", "glaser", "adapt"]
BASELINES = ["chen", "yeung", "kramberger", "glaser"]
ADAPTIVE = [*BASELINES, "adapt"]
VARIANTS = {"harmonica": "HarmonICA", "harmonica_theta": "HarmonICA (θ)"}
THRESHOLDS = (0.99, 0.95, 0.90, 0.80)
COLORS = {
    "harmonica": "#1f77b4",
    "ica": "#444444",
    "chen": "#e7298a",
    "yeung": "#a6761d",
    "kramberger": "#66a61e",
    "glaser": "#9e7c4d",
    "adapt": "#1b9e77",
}
LABELS = {
    "harmonica": "HarmonICA (ours)",
    "ica": "ICA / cBSS",
    "chen": "Chen 2020",
    "yeung": "Yeung 2024",
    "kramberger": "Kramberger 2021",
    "glaser": "Glaser 2018",
    "adapt": "Mendez Guerra 2024",
    "harmonica_theta": "HarmonICA (θ)",
    "ica_harness": "ICA / cBSS (Adapt harness)",
}
LINEWIDTH = {"harmonica": 4.8, "ica": 3.0}
ZORDER = {"harmonica": 5, "ica": 4}
# Annotation text anchors (data coordinates), chosen to sit in empty space.
OURS_TEXT = (4.6, 0.912)
ICA_TEXT = {"recall": (6.3, 0.752), "f1": (2.3, 0.17)}
LEGEND_ANCHOR = (12.0, 0.765)
METRIC_NAME = {"recall": "Recall", "f1": "F1"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def build(inputs: dict[str, list[str]], axis: str, denominator: str) -> dict[str, dict[str, dict]]:
    """``{grid: {method: {"values", "denom", "seeds", "frame"}}}``."""
    col = "r" if axis == "recall" else "f1"
    loaded = {m: read_units(globs, ica=(m == "ica")) for m, globs in inputs.items()}
    grids = sorted({g for units, _ in loaded.values() for g in units["grid"].unique()})
    n_rec = {g: len({d for _, ev in loaded.values() for d in ev.loc[ev.grid == g, "dataset"]})
             for g in grids}
    out: dict[str, dict[str, dict]] = {g: {} for g in grids}
    for m, (units, ev) in loaded.items():
        for g in grids:
            u, e = units[units.grid == g], ev[ev.grid == g]
            if e.empty:
                continue
            u = u.dropna(subset=["r", "f1"])
            u = u.sort_values("f1", ascending=False, kind="mergesort").drop_duplicates(
                ["dataset", "seed", "gt_mu"])
            u = u[u["f1"] > 0]
            den = len(e) if denominator == "matched" else n_rec[g] * e["seed"].nunique()
            out[g][m] = {"values": u[col].to_numpy(dtype=float), "denom": den, "frame": u}
    return out


def curve(values: np.ndarray, denom: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted-descending curve: (x = rank / denom, y = value)."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    return np.arange(1, len(v) + 1) / denom, v


def crossover(values: np.ndarray, denom: int, thr: float) -> float:
    """Mean units per recording-seed with value >= thr."""
    return float((np.asarray(values) >= thr).sum()) / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, curves: dict, axis: str, *, annotate: bool, legend: bool, thr_labels: bool) -> None:
    name = METRIC_NAME[axis]
    for m in ORDER:
        if m not in curves:
            continue
        x, v = curve(curves[m]["values"], curves[m]["denom"])
        z = ZORDER.get(m, 2)
        ax.plot(x, v, color=COLORS[m], lw=LINEWIDTH.get(m, 2.0), ls="--" if m == "ica" else "-",
                zorder=z, label=LABELS[m], solid_capstyle="butt")
        ax.scatter([x[-1]], [v[-1]], color=COLORS[m], s=36, zorder=z + 1,
                   edgecolor="white", linewidth=0.8)
    for thr in THRESHOLDS:
        ax.axhline(thr, color="black", lw=0.7, ls=":", alpha=0.55, zorder=1)
    if thr_labels:
        # 0.99 sits above its line; the others are centred on theirs on a white box.
        for thr, va, dy in ((0.99, "bottom", 0.003), (0.95, "center", 0.0),
                            (0.90, "center", 0.0), (0.80, "center", 0.0)):
            ax.text(11.95, thr + dy, f"{name}={thr:.2f}", fontsize=11, color="dimgray",
                    ha="right", va=va, zorder=3,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9)
                    if va == "center" else None)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("MU rank, normalised per recording-seed")
    ax.grid(alpha=0.28, zorder=0)
    if annotate and {"harmonica", "ica"} <= set(curves):
        x_ours = crossover(curves["harmonica"]["values"], curves["harmonica"]["denom"], 0.95)
        x_ica = crossover(curves["ica"]["values"], curves["ica"]["denom"], 0.95)
        ax.annotate(
            f"HarmonICA (ours) holds {name}$\\,\\geq\\,$0.95\n"
            f"until ~{x_ours:.1f} MUs/rec  ({x_ours / x_ica:.1f}× ICA)",
            xy=(x_ours, 0.95), xytext=OURS_TEXT, fontsize=13, color=COLORS["harmonica"],
            fontweight="bold", va="center", ha="left", zorder=7,
            arrowprops=dict(arrowstyle="->", color=COLORS["harmonica"], lw=1.0, shrinkA=2, shrinkB=2),
        )
        ax.annotate(
            f"ICA < {name}=0.95\nat ~{x_ica:.1f} MUs/rec",
            xy=(x_ica, 0.95), xytext=ICA_TEXT[axis], fontsize=13, color="black", va="center",
            ha="left", zorder=7,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85)
            if axis == "f1" else None,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.9, shrinkA=2, shrinkB=2,
                            connectionstyle="arc3,rad=0.0"),
        )
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=LEGEND_ANCHOR, bbox_transform=ax.transData,
                  ncol=1, fontsize=11, framealpha=0.95, handlelength=2.2, borderaxespad=0.0,
                  labelspacing=0.35)


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig] {stem}.{{pdf,png}}")


def fig_single(curves: dict, axis: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.22, 4.71))
    draw_panel(ax, curves, axis, annotate=True, legend=True, thr_labels=True)
    ax.set_ylabel(f"{METRIC_NAME[axis]} of rank-$K$ MU")
    fig.tight_layout()
    _save(fig, stem)


def fig_combined(all_curves: dict, axis: str, stem: Path) -> None:
    grids = sorted(all_curves)
    fig, axes = plt.subplots(1, len(grids), figsize=(15.0, 4.85), sharey=True)
    for ax, g in zip(axes, grids):
        draw_panel(ax, all_curves[g], axis, annotate=False, legend=False, thr_labels=(g == grids[-1]))
        ax.set_title(g, fontsize=15)
    axes[0].set_ylabel(f"{METRIC_NAME[axis]} of rank-$K$ MU")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=12, handlelength=2.2, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

FULL_T = np.round(np.arange(0.001, 1.0001, 0.001), 3)
SWEEP_T = np.round(np.arange(0.50, 1.0001, 0.005), 3)


def _yields(c: dict, ts: np.ndarray) -> np.ndarray:
    v = np.sort(c["values"])
    return (len(v) - np.searchsorted(v, ts, side="left")) / c["denom"]


def _ranges(ts: np.ndarray) -> str:
    if len(ts) == 0:
        return "none"
    out, start, prev = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - prev > 0.0015:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return ", ".join(f"[{a:.3f},{b:.3f}]" for a, b in out)


def exact_min_gap(a: dict, b: dict) -> tuple[float, float]:
    """Min over every threshold in (0, 1] of yield_a - yield_b, and where.

    Both yields are step functions that change only at data values, so the
    union of the two value sets covers every threshold.
    """
    ts = np.unique(np.concatenate([a["values"], b["values"]]))
    ts = ts[(ts > 0) & (ts <= 1)]
    d = _yields(a, ts) - _yields(b, ts)
    i = int(np.argmin(d))
    return float(d[i]), float(ts[i])


def crossover_table(all_curves: dict, axis: str) -> pd.DataFrame:
    name = METRIC_NAME[axis]
    rows = []
    for g, curves in all_curves.items():
        for m, c in curves.items():
            rec = {"grid": g, "method": m, "label": LABELS.get(m, m),
                   "denominator_recseeds": c["denom"], "n_matched_units": len(c["values"]),
                   "coverage_cap": len(c["values"]) / c["denom"]}
            for t in THRESHOLDS:
                rec[f"{name}>={t:.2f}"] = crossover(c["values"], c["denom"], t)
            rows.append(rec)
    return pd.DataFrame(rows)


def sweep_table(all_curves: dict) -> pd.DataFrame:
    rows = []
    for g, curves in all_curves.items():
        ys = {m: _yields(c, SWEEP_T) for m, c in curves.items()}
        for i, t in enumerate(SWEEP_T):
            rows.append({"grid": g, "T": t, **{m: y[i] for m, y in ys.items()}})
    return pd.DataFrame(rows)


def dominance_text(all_curves: dict, axis: str) -> str:
    name = METRIC_NAME[axis]
    lines = []
    for g, curves in all_curves.items():
        variants = [h for h in VARIANTS if h in curves]
        if not variants:
            continue
        Y = {m: _yields(c, FULL_T) for m, c in curves.items()}
        lines.append(f"[{g}] threshold T swept over (0,1], step 0.001; "
                     f"y = MUs per recording-seed with {name} >= T")
        others = [m for m in ORDER if m in curves and m != "harmonica"]
        for h in variants:
            lab = VARIANTS[h]
            for o in others:
                d = Y[h] - Y[o]
                lines.append(f"    {lab} - {LABELS[o]:<18s}: min {d.min():.3f} at T={FULL_T[d.argmin()]:.3f}; "
                             f"{lab} > it at every T: {bool((d > 0).all())}; "
                             f"{lab} <= it for T in {_ranges(FULL_T[d <= 0])}")
            gaps = {}
            for o in others:
                gap, t = exact_min_gap(curves[h], curves[o])
                gaps[o] = gap
                lines.append(f"    EXACT (all data thresholds) {lab} - {LABELS[o]:<18s}: "
                             f"min {gap:.4f} at T={t:.4f}; > 0: {gap > 0}")
            adaptive = [o for o in others if o in ADAPTIVE]
            if adaptive:
                worst = min(adaptive, key=gaps.get)
                lines.append(f"    {lab} above every adaptive baseline at every T: "
                             f"{all(gaps[o] > 0 for o in adaptive)} (smallest margin {gaps[worst]:.4f}, "
                             f"vs {LABELS[worst]})")
        if "adapt" in curves and "ica" in curves:
            da = Y["adapt"] - Y["ica"]
            lines.append(f"    {LABELS['adapt']} > ICA for T in {_ranges(FULL_T[da > 0])}; "
                         f"< ICA for T in {_ranges(FULL_T[da < 0])}")
            rest = [b for b in BASELINES if b in curves]
            if rest:
                beats = Y["adapt"] > np.max([Y[b] for b in rest], axis=0)
                lines.append(f"    {LABELS['adapt']} > every other adaptive baseline for T in "
                             f"{_ranges(FULL_T[beats])}")
        for t in THRESHOLDS:
            vals = {m: crossover(c["values"], c["denom"], t) for m, c in curves.items()}
            lines.append(f"    {name}>={t:.2f}: " + " | ".join(f"{LABELS.get(m, m)} {v:.3f}"
                                                         for m, v in vals.items()))
        if "adapt" in curves and "harmonica" in curves:
            # unpooled check on the seeds both methods share
            col = "r" if axis == "recall" else "f1"
            h, a = curves["harmonica"]["frame"], curves["adapt"]["frame"]
            seeds = sorted(set(a["seed"]) & set(h["seed"]))
            for t in THRESHOLDS:
                hs = h[h[col] >= t].groupby("seed").size().reindex(seeds, fill_value=0)
                as_ = a[a[col] >= t].groupby("seed").size().reindex(seeds, fill_value=0)
                lines.append(f"    per seed, {name}>={t:.2f}: HarmonICA {hs.tolist()} vs "
                             f"{LABELS['adapt']} {as_.tolist()} -> HarmonICA higher in "
                             f"{(hs > as_).sum()}/{len(seeds)} seeds")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--axis", choices=["recall", "f1"], default="recall")
    ap.add_argument("--input", nargs="+", action="append", required=True,
                    metavar=("METHOD", "GLOB"), help="method name and its result CSVs (repeatable)")
    ap.add_argument("--denominator", choices=["matched", "all"], default="matched",
                    help="x normalisation: (recording, seed) pairs with rows (paper), "
                         "or recordings x seeds")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_curves = build(parse_inputs(args.input), args.axis, args.denominator)
    stem = f"pareto_{args.axis}"
    for g, curves in all_curves.items():
        fig_single(curves, args.axis, args.out_dir / f"{stem}_{g}")
    if len(all_curves) > 1:
        fig_combined(all_curves, args.axis, args.out_dir / f"{stem}_combined")
    tab = crossover_table(all_curves, args.axis)
    tab.to_csv(args.out_dir / f"{stem}_crossovers.csv", index=False)
    sweep_table(all_curves).to_csv(args.out_dir / f"{stem}_sweep.csv", index=False)
    txt = dominance_text(all_curves, args.axis)
    (args.out_dir / f"{stem}_dominance.txt").write_text(txt + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
