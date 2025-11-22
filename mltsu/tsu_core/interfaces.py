"""
Core interfaces for TSU backends
Defines the protocol that all TSU implementations must follow
"""

from typing import Protocol, Dict, Optional, Any, Tuple, Callable, runtime_checkable
import numpy as np
from dataclasses import dataclass


@dataclass
class TSUConfig:
    """Configuration for TSU backend operations"""

    beta: float = 1.0  # Inverse temperature (1/T)
    num_steps: int = 100  # Number of sampling steps
    seed: Optional[int] = None  # Random seed for reproducibility
    device: str = "cpu"  # Device: "cpu", "cuda", "tpu", "tsu"
    batch_size: int = 1  # Default batch size

    # Advanced parameters
    burnin_steps: int = 0  # Number of burn-in steps before sampling
    thinning: int = 1  # Keep every nth sample
    parallel_chains: int = 1  # Number of parallel MCMC chains

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {self.num_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")


@dataclass
class TSUSample:
    """Container for TSU sampling results"""

    samples: np.ndarray  # Sampled states
    energies: Optional[np.ndarray] = None  # Energy of each sample
    acceptance_rate: Optional[float] = None  # MCMC acceptance rate
    trajectory: Optional[np.ndarray] = None  # Full sampling trajectory if requested
    metadata: Dict[str, Any] = None  # Additional backend-specific metadata

    def __post_init__(self):
        """Initialize metadata if not provided"""
        if self.metadata is None:
            self.metadata = {}


@runtime_checkable
class TSUBackend(Protocol):
    """
    Protocol defining the interface for all TSU backend implementations.
    This allows swapping between simulators and real hardware seamlessly.
    """

    def sample_ising(
        self,
        J: np.ndarray,  # Coupling matrix (N, N)
        h: np.ndarray,  # External field (N,)
        beta: float,  # Inverse temperature
        num_steps: int,  # Number of sampling steps
        batch_size: int = 1,  # Number of independent samples
        init_state: Optional[np.ndarray] = None,  # Initial state
        record_trajectory: bool = False,  # Record full trajectory
        key: Optional[Any] = None  # Random key/seed
    ) -> Dict[str, np.ndarray]:
        """
        Sample from an Ising model with Hamiltonian:
        H(s) = -0.5 * s^T J s - h^T s

        where s ∈ {-1, +1}^N

        The probability distribution is:
        P(s) ∝ exp(-β * H(s))

        Args:
            J: Symmetric coupling matrix of shape (N, N)
            h: External field vector of shape (N,)
            beta: Inverse temperature (higher = lower temperature)
            num_steps: Number of MCMC steps
            batch_size: Number of independent samples to generate
            init_state: Initial spin configuration (optional)
            record_trajectory: If True, return full sampling trajectory
            key: Random seed or key for reproducibility

        Returns:
            Dictionary containing:
                - 'samples': Final samples of shape (batch_size, N)
                - 'final_energy': Energy of final samples (batch_size,)
                - 'trajectory': Full trajectory if requested (batch_size, num_steps, N)
                - 'acceptance_rate': MCMC acceptance rate (if applicable)
        """
        ...

    def sample_binary_layer(
        self,
        logits: np.ndarray,  # Input logits (batch, n_bits)
        beta: float = 1.0,  # Temperature parameter
        num_steps: int = 1,  # Number of refinement steps
        key: Optional[Any] = None  # Random key
    ) -> np.ndarray:
        """
        Sample binary states from independent or weakly coupled units.

        For independent units with logits ℓᵢ:
        P(xᵢ = 1) = σ(β * ℓᵢ) where σ is sigmoid

        Args:
            logits: Input logits of shape (batch_size, n_bits)
            beta: Inverse temperature (controls sharpness)
            num_steps: Number of refinement steps (for coupled case)
            key: Random seed or key

        Returns:
            Binary samples of shape (batch_size, n_bits) with values in {0, 1}
        """
        ...

    def sample_custom(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        init_state: np.ndarray,
        num_steps: int,
        beta: float = 1.0,
        key: Optional[Any] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generic energy-based sampling for custom energy functions.

        Samples from P(x) ∝ exp(-β * E(x))

        Args:
            energy_fn: Function that computes energy for given state
            init_state: Initial state configuration
            num_steps: Number of sampling steps
            beta: Inverse temperature
            key: Random seed or key
            **kwargs: Additional backend-specific parameters

        Returns:
            Sampled states
        """
        ...

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the backend capabilities and configuration.

        Returns:
            Dictionary with backend information:
                - 'name': Backend name
                - 'type': 'simulator' or 'hardware'
                - 'capabilities': List of supported operations
                - 'max_bits': Maximum number of bits/spins
                - 'device': Device information
        """
        ...

    def reset(self) -> None:
        """Reset the backend state (clear caches, reset random seeds, etc.)"""
        ...


class TSUBackendRegistry:
    """Registry for TSU backend implementations"""

    _backends: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a new TSU backend implementation"""
        if not issubclass(backend_class, TSUBackend):
            raise TypeError(f"{backend_class} must implement TSUBackend protocol")
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered backend class by name"""
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> list:
        """List all registered backend names"""
        return list(cls._backends.keys())