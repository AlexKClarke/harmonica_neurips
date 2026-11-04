"""HarmonICA and the baseline EMG-decomposition methods it is compared against.

Every runner shares one contract:

.. code-block:: python

    spikes = run_<method>(emg, FS, **kwargs)   # -> dict[int, np.ndarray]

``emg`` is channels-first ``(n_channels, n_samples)``. Matching against
ground truth happens externally via
:func:`harmonica.evaluate.evaluate_decomposition`.

- :mod:`.signal_space` — **HarmonICA** (the paper's method): FastICA
  warm start, then a low-rank time-varying compensator trained jointly
  with the separation vector, with neural peel-off between units.
- :mod:`.glaser`, :mod:`.glaser_cyclostationary` — Glaser & Holobar 2018
- :mod:`.kramberger` — Kramberger & Holobar 2021 (needs the joint angle)
- :mod:`.chen`       — Chen et al. 2020 (adaptive CKC)
- :mod:`.yeung`      — Yeung et al. 2024 (directional-forgetting RLS)
- :mod:`.adapt`      — Mendez Guerra et al. 2024, adapt_decomp v0 engine
- :mod:`.adapt_v1`   — Mendez Guerra et al. 2024, adapt_decomp v1.0.0 engine
  (the paper's Adapt row; needs the ``[adapt]`` extra, so it is imported
  lazily on first use of :func:`run_adapt_v1`)
- :mod:`.tcl`, :mod:`.ivae` — identifiable nonlinear ICA (TCL, iVAE)
"""
from __future__ import annotations

from .adapt import run_adapt
from .chen import run_chen
from .glaser import run_glaser
from .glaser_cyclostationary import run_glaser_cyclostationary
from .ivae import run_ivae
from .kramberger import run_kramberger
from .signal_space import run_signal_space
from .tcl import run_tcl
from .yeung import run_yeung


def __getattr__(name: str):
    # run_adapt_v1 pulls in the vendored v1 engine and its dependencies
    # (optuna, plotly, ...); import it only when asked for.
    if name == "run_adapt_v1":
        from .adapt_v1 import run_adapt_v1
        return run_adapt_v1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_adapt",
    "run_adapt_v1",
    "run_chen",
    "run_glaser",
    "run_glaser_cyclostationary",
    "run_ivae",
    "run_kramberger",
    "run_signal_space",
    "run_tcl",
    "run_yeung",
]
