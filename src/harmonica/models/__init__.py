"""Building blocks behind HarmonICA and the nonlinear-ICA baselines.

- :mod:`.engine` — PyTorch HarmonICA engine (compensator networks,
  joint separation / compensation step, spike timestamping). Used by
  :func:`harmonica.baselines.run_signal_space`.
- :mod:`.harmonica_signal_space` — ``refine_signal_space_comp_warmstart``,
  the per-unit form of HarmonICA (refines one warm-started filter, no
  peel-off). Used by ``scripts/bench_sweep.py`` and the per-unit ablations.
- :mod:`.tcl`, :mod:`.ivae` — TCL and iVAE implementations used by the
  nonlinear-ICA baselines.
- :mod:`.guardrails_zoo` — ``proj_ratio_guardrail`` (used by
  :func:`harmonica.nonstationary.pipelines.run_all_methods`).
"""
from .guardrails_zoo import proj_ratio_guardrail
from .harmonica_signal_space import refine_signal_space_comp_warmstart
from .ivae import (
    iVAE,
    extract_sources,
    ivae_decompose,
    make_combined_labels,
    make_pose_labels,
    make_segment_labels,
    train_ivae,
)
from .tcl import DeepTCL, deep_tcl_decompose, segment_signal

__all__ = [
    "DeepTCL",
    "deep_tcl_decompose",
    "extract_sources",
    "iVAE",
    "ivae_decompose",
    "make_combined_labels",
    "make_pose_labels",
    "make_segment_labels",
    "proj_ratio_guardrail",
    "refine_signal_space_comp_warmstart",
    "segment_signal",
    "train_ivae",
]
