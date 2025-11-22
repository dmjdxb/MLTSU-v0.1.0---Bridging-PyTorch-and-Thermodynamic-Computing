"""
PyTorch integration for TSU components
Provides PyTorch modules that use TSU backends for sampling
"""

from .bridge import torch_to_jax, jax_to_torch, numpy_to_torch
from .binary_layer import TSUBinaryLayer, StraightThroughEstimator
from .noise import TSUGaussianNoise

__all__ = [
    "torch_to_jax",
    "jax_to_torch",
    "numpy_to_torch",
    "TSUBinaryLayer",
    "StraightThroughEstimator",
    "TSUGaussianNoise",
]