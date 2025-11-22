"""
MLTSU - Machine Learning Thermodynamic Sampling Units
A PyTorch-native framework for bridging deep learning with thermodynamic computing hardware
"""

__version__ = "0.1.0"

# Core interfaces
from .tsu_core.interfaces import TSUBackend, TSUConfig

__all__ = [
    "TSUBackend",
    "TSUConfig",
]