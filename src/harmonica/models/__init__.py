"""Building blocks behind HarmonICA and the nonlinear-ICA baselines.

- :mod:`.engine` — PyTorch HarmonICA engine (compensator networks,
  joint separation / compensation step, spike timestamping). Used by
  :func:`harmonica.baselines.run_signal_space`.
- :mod:`.harmonica_signal_space` — ``refine_signal_space_comp_warmstart``,
  the per-unit form of HarmonICA (refines one warm-started filter, no
  peel-off). Used by ``scripts/bench_sweep.py`` for the no-peel-off ablation.
- :mod:`.tcl`, :mod:`.ivae` — TCL and iVAE implementations used by
  :func:`harmonica.baselines.run_tcl` and :func:`harmonica.baselines.run_ivae`.
"""
from .harmonica_signal_space import refine_signal_space_comp_warmstart

__all__ = [
    "refine_signal_space_comp_warmstart",
]
