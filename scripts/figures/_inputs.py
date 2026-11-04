"""Per-unit result CSVs -> one frame, shared by the figure and statistics scripts.

Two layouts are read:

- one CSV per seed, from ``bench_baselines.py`` / ``bench_sweep.py``, named
  ``<grid>_<tag>_seed<k>.csv`` (``ch070_chen_seed0.csv``): one row per matched
  ground-truth unit, columns ``dataset``, ``gt_mu``, ``n_gt``, ``f1``, ``r``
  (``f1_ica`` / ``r_ica`` for the unrefined FastICA source of a sweep);
- the long format of ``bench_adapt.py``: one row per ground-truth unit and
  seed, every seed in one file, columns ``grid``, ``recording``, ``seed_idx``,
  ``gt_id``, ``best_f1``, ``best_recall``, ``ok``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

COLUMNS = ["grid", "dataset", "seed", "gt_mu", "n_gt", "f1", "r"]


def expand(patterns: list[str]) -> list[Path]:
    """Sorted, de-duplicated paths matched by shell-style globs.

    Raises:
        FileNotFoundError: If a pattern matches nothing.
    """
    out = set()
    for pat in patterns:
        hits = glob.glob(pat)
        if not hits:
            raise FileNotFoundError(f"no file matches {pat}")
        out.update(Path(h) for h in hits)
    return sorted(out)


def read_units(patterns: list[str], ica: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one method's per-unit results.

    Args:
        patterns: Globs of result CSVs (either layout; may be mixed).
        ica: Score the unrefined FastICA source (``f1_ica`` / ``r_ica``) of
            per-seed sweep CSVs.

    Returns:
        ``(units, evaluated)``. ``units`` has one row per input row (no
        de-duplication; unmatched long-format rows have F1 0), columns
        ``grid, dataset, seed, gt_mu, n_gt, f1, r``. ``evaluated`` lists the
        ``(grid, dataset, seed)`` triples the inputs hold any row for.
    """
    units, evaluated = [], []
    for path in expand(patterns):
        df = pd.read_csv(path)
        if {"seed_idx", "best_f1"} <= set(df.columns):
            if df["method"].nunique() > 1:
                raise SystemExit(f"{path} holds several methods {sorted(df['method'].unique())}; "
                                 "split it by method first")
            df = df.assign(grid=[f"ch{int(g):03d}" for g in df["grid"]])
            df = df.rename(columns={"recording": "dataset", "seed_idx": "seed", "gt_id": "gt_mu",
                                    "best_f1": "f1", "best_recall": "r"})
            evaluated.append(df[["grid", "dataset", "seed"]])
            units.append(df[df["ok"].astype(bool)][COLUMNS])
            continue
        m_grid = re.match(r"(ch\d{3})_", path.name)
        m_seed = re.search(r"_seed(\d+)\.csv$", path.name)
        if not (m_grid and m_seed):
            raise SystemExit(f"{path.name}: expected <grid>_<tag>_seed<k>.csv")
        df = df.assign(grid=m_grid.group(1), seed=int(m_seed.group(1)))
        if ica:
            df = df.drop(columns=["f1", "r"]).rename(columns={"f1_ica": "f1", "r_ica": "r"})
        evaluated.append(df[["grid", "dataset", "seed"]])
        units.append(df[COLUMNS])
    return (pd.concat(units, ignore_index=True),
            pd.concat(evaluated, ignore_index=True).drop_duplicates().reset_index(drop=True))


def parse_inputs(specs: list[list[str]]) -> dict[str, list[str]]:
    """``--input METHOD GLOB [GLOB ...]`` (repeatable) -> ``{method: [globs]}``."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        if len(spec) < 2:
            raise SystemExit(f"--input needs METHOD and at least one glob, got {spec}")
        out.setdefault(spec[0], []).extend(spec[1:])
    return out
