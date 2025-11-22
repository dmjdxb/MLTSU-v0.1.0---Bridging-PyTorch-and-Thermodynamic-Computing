"""
JAX-based TSU backend implementation
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Any, Callable
from ..tsu_core.interfaces import TSUBackend, TSUBackendRegistry
from .sampler import (
    gibbs_sample_ising,
    metropolis_sample_ising,
    parallel_tempering_ising,
    langevin_sample,
)
from .energy_models import ising_energy, binary_layer_energy


class JAXTSUBackend:
    """
    JAX-based software simulator for TSU operations.
    Implements the TSUBackend protocol using JAX for fast vectorized sampling.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        device: str = "cpu",
        sampling_method: str = "gibbs",
    ):
        """
        Initialize JAX TSU backend.

        Args:
            seed: Random seed for reproducibility
            device: Device to run on ('cpu', 'gpu', 'tpu')
            sampling_method: Default sampling method ('gibbs', 'metropolis', 'parallel_tempering')
        """
        self.key = jax.random.PRNGKey(seed or 0)
        self.device = device
        self.sampling_method = sampling_method

        # Set JAX device
        if device == "gpu" and jax.devices("gpu"):
            self._device = jax.devices("gpu")[0]
        elif device == "tpu" and jax.devices("tpu"):
            self._device = jax.devices("tpu")[0]
        else:
            self._device = jax.devices("cpu")[0]

    def sample_ising(
        self,
        J: np.ndarray,
        h: np.ndarray,
        beta: float,
        num_steps: int,
        batch_size: int = 1,
        init_state: Optional[np.ndarray] = None,
        record_trajectory: bool = False,
        key: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample from Ising model using selected sampling method.

        Args:
            J: Coupling matrix (N, N)
            h: External field (N,)
            beta: Inverse temperature
            num_steps: Number of sampling steps
            batch_size: Number of independent samples
            init_state: Initial state (optional)
            record_trajectory: Record full sampling trajectory
            key: Random key (optional)

        Returns:
            Dictionary with samples and metadata
        """
        # Convert to JAX arrays
        J_jax = jnp.array(J, dtype=jnp.float32)
        h_jax = jnp.array(h, dtype=jnp.float32)

        # Ensure J is symmetric
        J_jax = (J_jax + J_jax.T) / 2

        # Get random key
        if key is None:
            self.key, key = jax.random.split(self.key)

        # Prepare initial states if provided
        if init_state is not None:
            init_state_jax = jnp.array(init_state, dtype=jnp.float32)
        else:
            init_state_jax = None

        # Sample batch
        samples = []
        energies = []
        acceptance_rates = []
        trajectories = [] if record_trajectory else None

        for i in range(batch_size):
            key, subkey = jax.random.split(key)

            # Choose sampling method
            if self.sampling_method == "gibbs":
                sample, info = gibbs_sample_ising(
                    J_jax, h_jax, beta, num_steps, subkey,
                    init_state_jax, record_trajectory
                )
            elif self.sampling_method == "metropolis":
                sample, info = metropolis_sample_ising(
                    J_jax, h_jax, beta, num_steps, subkey, init_state_jax
                )
            elif self.sampling_method == "parallel_tempering":
                # Use geometric temperature ladder
                betas = beta * jnp.logspace(-0.5, 0.5, 8)
                states, info = parallel_tempering_ising(
                    J_jax, h_jax, betas, num_steps, subkey
                )
                sample = states[0]  # Return coldest replica
                info["energy"] = info["energies"][0]
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")

            samples.append(np.array(sample))
            energies.append(float(info["energy"]))

            if "acceptance_rate" in info:
                acceptance_rates.append(float(info["acceptance_rate"]))

            if record_trajectory and info.get("trajectory") is not None:
                trajectories.append(np.array(info["trajectory"]))

        # Prepare return dictionary
        result = {
            "samples": np.stack(samples),
            "final_energy": np.array(energies),
        }

        if acceptance_rates:
            result["acceptance_rate"] = np.mean(acceptance_rates)

        if trajectories:
            result["trajectory"] = np.stack(trajectories)

        return result

    def sample_binary_layer(
        self,
        logits: np.ndarray,
        beta: float = 1.0,
        num_steps: int = 1,
        key: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Sample binary states from logits.

        For independent units:
        P(x_i = 1) = sigmoid(beta * logit_i)

        Args:
            logits: Input logits (batch, n_bits)
            beta: Inverse temperature
            num_steps: Number of refinement steps (for future coupled implementations)
            key: Random key

        Returns:
            Binary samples (batch, n_bits) with values in {0, 1}
        """
        logits_jax = jnp.array(logits, dtype=jnp.float32)

        if key is None:
            self.key, key = jax.random.split(self.key)

        # Compute probabilities
        probs = jax.nn.sigmoid(beta * logits_jax)

        # Sample binary states
        key, subkey = jax.random.split(key)
        samples = jax.random.bernoulli(subkey, probs)

        # If num_steps > 1, could implement iterative refinement here
        # For now, just return independent samples

        return np.array(samples, dtype=np.float32)

    def sample_custom(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        init_state: np.ndarray,
        num_steps: int,
        beta: float = 1.0,
        key: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Sample from custom energy function using Langevin dynamics.

        Args:
            energy_fn: Energy function E(x)
            init_state: Initial state
            num_steps: Number of sampling steps
            beta: Inverse temperature
            key: Random key
            **kwargs: Additional parameters (e.g., step_size)

        Returns:
            Sampled state
        """
        init_state_jax = jnp.array(init_state, dtype=jnp.float32)

        if key is None:
            self.key, key = jax.random.split(self.key)

        # Convert energy function to JAX
        def jax_energy_fn(x):
            return jnp.array(energy_fn(x))

        # Get step size
        step_size = kwargs.get("step_size", 0.01)

        # Run Langevin sampling
        final_state, info = langevin_sample(
            jax_energy_fn, init_state_jax, beta, num_steps, step_size, key
        )

        return np.array(final_state)

    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information.

        Returns:
            Dictionary with backend capabilities and configuration
        """
        return {
            "name": "JAXTSUBackend",
            "type": "simulator",
            "device": str(self._device),
            "sampling_method": self.sampling_method,
            "capabilities": [
                "ising",
                "binary_layer",
                "custom_energy",
                "parallel_tempering",
                "trajectory_recording",
            ],
            "max_bits": 10000,  # Practical limit for simulation
            "jax_version": jax.__version__,
            "has_gpu": len(jax.devices("gpu")) > 0,
            "has_tpu": len(jax.devices("tpu")) > 0,
        }

    def reset(self) -> None:
        """Reset the backend state."""
        # Generate new random key
        self.key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    def benchmark_sampling_speed(
        self, n_spins: int = 100, num_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Benchmark sampling speed for different methods.

        Args:
            n_spins: Number of spins
            num_steps: Number of sampling steps

        Returns:
            Dictionary with timing results
        """
        import time

        # Create random Ising model
        key = jax.random.PRNGKey(42)
        key_J, key_h = jax.random.split(key)
        J = jax.random.normal(key_J, (n_spins, n_spins))
        J = (J + J.T) / 2
        h = jax.random.normal(key_h, (n_spins,))

        results = {}

        # Benchmark Gibbs sampling
        start = time.time()
        _ = gibbs_sample_ising(J, h, 1.0, num_steps, key)
        results["gibbs"] = time.time() - start

        # Benchmark Metropolis sampling
        start = time.time()
        _ = metropolis_sample_ising(J, h, 1.0, num_steps, key)
        results["metropolis"] = time.time() - start

        # Samples per second
        for method in results:
            results[f"{method}_samples_per_sec"] = num_steps / results[method]

        return results


# Register the backend
TSUBackendRegistry.register("jax", JAXTSUBackend)