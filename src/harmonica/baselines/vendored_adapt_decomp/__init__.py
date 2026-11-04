"""Vendored copy of ``imendezguerra/adapt_decomp`` (MIT licensed).

Source: https://github.com/imendezguerra/adapt_decomp
Pinned commit: ``8ffeb66ebd30bab9b5167a3fc485c2330a81da0f`` (2026-03 branch HEAD)
Original license: see ``LICENSE_adapt_decomp`` in this directory.

Modifications from upstream: absolute imports
(``from adapt_decomp.config import Config``) rewritten to relative
(``from .config import Config``) so the package can be imported without
being installed at the top level; the unused ``loaders.py``, ``plots.py``
and ``utils.py`` are not vendored.

Cite as: Mendez Guerra et al., *Adaptive EMG decomposition in dynamic
conditions based on online learning metrics with tunable
hyperparameters*, J Neural Eng 21(4), 2024.
DOI: ``10.1088/1741-2552/ad5ebf``.
"""
