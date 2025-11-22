"""
State representations for p-bit networks
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional


class PBitState(NamedTuple):
    """State of a p-bit network"""

    spins: jnp.ndarray  # Current spin configuration {-1, +1}^N
    energy: jnp.ndarray  # Current energy (scalar)
    key: jax.random.PRNGKey  # Random state
    step: jnp.ndarray  # Current step number (scalar)
    acceptance_rate: jnp.ndarray  # Running acceptance rate (scalar)


def init_pbit_state(
    n_bits: int,
    key: jax.random.PRNGKey,
    init_spins: Optional[jnp.ndarray] = None,
    init_energy: Optional[float] = None,
) -> PBitState:
    """
    Initialize a p-bit state.

    Args:
        n_bits: Number of p-bits/spins
        key: JAX random key
        init_spins: Optional initial spin configuration
        init_energy: Optional initial energy

    Returns:
        Initialized PBitState
    """
    key, subkey = jax.random.split(key)

    if init_spins is None:
        # Random initialization
        spins = jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_bits,))
    else:
        spins = init_spins

    if init_energy is None:
        energy = jnp.array(0.0)
    else:
        energy = jnp.array(init_energy)

    return PBitState(
        spins=spins,
        energy=energy,
        key=key,
        step=jnp.array(0),
        acceptance_rate=jnp.array(1.0),
    )


def batch_init_pbit_state(
    batch_size: int,
    n_bits: int,
    key: jax.random.PRNGKey,
    init_spins: Optional[jnp.ndarray] = None,
) -> PBitState:
    """
    Initialize a batch of p-bit states.

    Args:
        batch_size: Number of independent states
        n_bits: Number of p-bits/spins per state
        key: JAX random key
        init_spins: Optional initial spin configurations (batch_size, n_bits)

    Returns:
        Batched PBitState
    """
    keys = jax.random.split(key, batch_size + 1)
    key = keys[0]
    subkeys = keys[1:]

    if init_spins is None:
        # Random initialization for each state in batch
        spins = jax.vmap(
            lambda k: jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(n_bits,))
        )(subkeys)
    else:
        spins = init_spins

    return PBitState(
        spins=spins,  # (batch_size, n_bits)
        energy=jnp.zeros(batch_size),
        key=key,
        step=jnp.array(0),
        acceptance_rate=jnp.ones(batch_size),
    )