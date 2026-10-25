"""
HarmonICA - EMG signal decomposition package
"""

from .configs import Config, IntramuscularConfig, SurfaceConfig
from .decomposition import DecompositionModule

__all__ = [
    "Config",
    "IntramuscularConfig",
    "SurfaceConfig",
    "DecompositionModule",
] 