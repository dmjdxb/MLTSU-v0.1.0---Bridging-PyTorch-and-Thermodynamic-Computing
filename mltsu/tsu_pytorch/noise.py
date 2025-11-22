"""
TSU-based noise generation for PyTorch
Uses p-bits and central limit theorem to generate approximate Gaussian noise
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from ..tsu_core.interfaces import TSUBackend
from .bridge import numpy_to_torch


class TSUGaussianNoise:
    """
    Generate approximate Gaussian noise using TSU/p-bits.

    Uses the central limit theorem: sum of M independent p-bits
    converges to Gaussian distribution as M increases.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        M: int = 12,
        beta: float = 1.0,
        num_steps: int = 1,
    ):
        """
        Initialize TSU Gaussian noise generator.

        Args:
            tsu_backend: TSU backend for sampling
            M: Number of p-bits per scalar (higher = better approximation)
            beta: Temperature parameter
            num_steps: Number of sampling steps
        """
        self.tsu_backend = tsu_backend
        self.M = M
        self.beta = beta
        self.num_steps = num_steps

        # Precompute normalization factor
        # For M unbiased p-bits mapped to {-1, +1}:
        # Mean = 0, Variance = M, so divide by sqrt(M) for unit variance
        self.normalization = 1.0 / np.sqrt(M)

    def sample(
        self,
        shape: Union[Tuple[int, ...], torch.Size],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Sample approximate Gaussian noise.

        Args:
            shape: Shape of noise tensor to generate
            device: Target device for output
            dtype: Target dtype for output

        Returns:
            Approximate Gaussian noise tensor ~N(0, 1)
        """
        # Calculate total number of scalars and p-bits needed
        if isinstance(shape, torch.Size):
            shape = tuple(shape)

        total_scalars = int(np.prod(shape))
        total_pbits = total_scalars * self.M

        # Create unbiased logits (P(bit=1) = 0.5)
        logits = np.zeros((1, total_pbits), dtype=np.float32)

        # Sample binary states from TSU
        binary_samples = self.tsu_backend.sample_binary_layer(
            logits, beta=self.beta, num_steps=self.num_steps
        )

        # Map {0, 1} -> {-1, +1}
        spins = 2 * binary_samples - 1

        # Reshape to group M p-bits per scalar
        spins = spins.reshape(total_scalars, self.M)

        # Sum M p-bits and normalize to get ~N(0, 1)
        gaussian = np.sum(spins, axis=1) * self.normalization

        # Reshape to target shape
        gaussian = gaussian.reshape(shape)

        # Convert to PyTorch tensor
        noise = numpy_to_torch(gaussian, device=device, dtype=dtype)

        return noise

    def sample_like(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample noise with same shape, device, and dtype as input tensor.

        Args:
            tensor: Reference tensor

        Returns:
            Noise tensor with matching properties
        """
        return self.sample(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )

    def validate_approximation(
        self, n_samples: int = 10000
    ) -> dict:
        """
        Validate that generated noise approximates N(0, 1).

        Args:
            n_samples: Number of samples to generate for validation

        Returns:
            Dictionary with statistical metrics
        """
        samples = self.sample((n_samples,))

        return {
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "min": float(samples.min()),
            "max": float(samples.max()),
            "mean_error": abs(float(samples.mean())),
            "std_error": abs(float(samples.std()) - 1.0),
        }


class TSUDiffusionNoise(TSUGaussianNoise):
    """
    Specialized noise generator for diffusion models.
    Includes variance scheduling and conditional noise generation.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        M: int = 12,
        beta: float = 1.0,
        num_steps: int = 1,
        schedule: str = "linear",
    ):
        """
        Initialize diffusion noise generator.

        Args:
            tsu_backend: TSU backend
            M: p-bits per scalar
            beta: Temperature
            num_steps: Sampling steps
            schedule: Noise schedule type ('linear', 'cosine', 'sigmoid')
        """
        super().__init__(tsu_backend, M, beta, num_steps)
        self.schedule = schedule

    def get_variance_schedule(
        self, timesteps: int
    ) -> torch.Tensor:
        """
        Get variance schedule for diffusion process.

        Args:
            timesteps: Number of diffusion timesteps

        Returns:
            Variance schedule tensor
        """
        if self.schedule == "linear":
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, timesteps)
        elif self.schedule == "cosine":
            steps = torch.arange(timesteps + 1)
            alpha_bar = torch.cos((steps / timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.schedule == "sigmoid":
            betas = torch.linspace(-6, 6, timesteps)
            return torch.sigmoid(betas) * (0.02 - 0.0001) + 0.0001
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def sample_timestep(
        self,
        x0: torch.Tensor,
        t: int,
        noise: Optional[torch.Tensor] = None,
        variance_schedule: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from x_0 using the forward diffusion process.

        Args:
            x0: Initial clean data
            t: Timestep
            noise: Optional pre-generated noise
            variance_schedule: Optional variance schedule

        Returns:
            Noisy sample x_t
        """
        if noise is None:
            noise = self.sample_like(x0)

        if variance_schedule is None:
            variance_schedule = self.get_variance_schedule(1000)

        # Get cumulative product of (1 - beta)
        alphas = 1.0 - variance_schedule
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Extract values for timestep t
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise


class TSUStructuredNoise:
    """
    Generate structured noise patterns using TSU.
    Useful for creating correlated noise, spatial patterns, etc.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        coupling_strength: float = 0.1,
    ):
        """
        Initialize structured noise generator.

        Args:
            tsu_backend: TSU backend
            coupling_strength: Strength of correlations between noise elements
        """
        self.tsu_backend = tsu_backend
        self.coupling_strength = coupling_strength

    def sample_correlated(
        self,
        shape: Tuple[int, ...],
        correlation_matrix: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample correlated noise using Ising-like coupling.

        Args:
            shape: Shape of noise tensor
            correlation_matrix: Desired correlation structure
            device: Target device

        Returns:
            Correlated noise tensor
        """
        total_dims = int(np.prod(shape))

        # Create coupling matrix for desired correlations
        if correlation_matrix is None:
            # Default: nearest-neighbor coupling for spatial correlation
            J = np.zeros((total_dims, total_dims))
            # Add simple nearest-neighbor connections
            for i in range(total_dims - 1):
                J[i, i + 1] = self.coupling_strength
                J[i + 1, i] = self.coupling_strength
        else:
            J = correlation_matrix * self.coupling_strength

        # No external field
        h = np.zeros(total_dims)

        # Sample from Ising model
        result = self.tsu_backend.sample_ising(
            J, h, beta=1.0, num_steps=100, batch_size=1
        )

        # Get samples and convert to continuous values
        samples = result["samples"][0]  # Shape: (total_dims,)

        # Add small Gaussian noise for continuity
        samples = samples + np.random.randn(*samples.shape) * 0.1

        # Reshape to target shape
        samples = samples.reshape(shape)

        # Convert to PyTorch
        return numpy_to_torch(samples, device=device)