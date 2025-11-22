"""
Energy functions for various thermodynamic models
"""

import jax
import jax.numpy as jnp
from typing import Optional, Callable


def ising_energy(
    spins: jnp.ndarray, J: jnp.ndarray, h: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Ising model energy.

    H(s) = -0.5 * s^T J s - h^T s

    Args:
        spins: Spin configuration {-1, +1}^N or (batch, N)
        J: Coupling matrix (N, N)
        h: External field (N,)

    Returns:
        Energy (scalar or batch of scalars)
    """
    if spins.ndim == 1:
        # Single configuration
        interaction = -0.5 * jnp.dot(spins, jnp.dot(J, spins))
        field = -jnp.dot(h, spins)
        return interaction + field
    else:
        # Batched configurations
        interaction = -0.5 * jnp.sum(spins @ J * spins, axis=1)
        field = -jnp.dot(spins, h)
        return interaction + field


def binary_layer_energy(
    states: jnp.ndarray, logits: jnp.ndarray
) -> jnp.ndarray:
    """
    Energy for independent binary units.

    E(x) = -sum_i x_i * logit_i

    Args:
        states: Binary states {0, 1}^N or (batch, N)
        logits: Logits (N,) or (batch, N)

    Returns:
        Energy (scalar or batch of scalars)
    """
    if states.ndim == 1:
        return -jnp.sum(states * logits)
    else:
        return -jnp.sum(states * logits, axis=-1)


def create_rbm_energy(
    W: jnp.ndarray, b_visible: jnp.ndarray, b_hidden: jnp.ndarray
) -> Callable:
    """
    Create energy function for Restricted Boltzmann Machine.

    E(v, h) = -v^T W h - b_v^T v - b_h^T h

    Args:
        W: Weight matrix (n_visible, n_hidden)
        b_visible: Visible bias (n_visible,)
        b_hidden: Hidden bias (n_hidden,)

    Returns:
        Energy function
    """

    def rbm_energy(state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute RBM energy for concatenated [visible, hidden] state
        """
        n_visible = W.shape[0]
        v = state[:n_visible]
        h = state[n_visible:]

        energy = -jnp.dot(v, jnp.dot(W, h))
        energy -= jnp.dot(b_visible, v)
        energy -= jnp.dot(b_hidden, h)
        return energy

    return rbm_energy


def create_potts_energy(J: jnp.ndarray, h: Optional[jnp.ndarray] = None) -> Callable:
    """
    Create energy function for Potts model (multi-state generalization of Ising).

    Args:
        J: Coupling tensor (N, N, q, q) where q is number of states
        h: External field (N, q)

    Returns:
        Energy function
    """
    N = J.shape[0]
    q = J.shape[2]

    def potts_energy(state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Potts energy for one-hot encoded state (N, q)
        """
        # Interaction term
        energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                energy -= jnp.dot(state[i], jnp.dot(J[i, j], state[j]))

        # Field term
        if h is not None:
            energy -= jnp.sum(state * h)

        return energy

    return potts_energy


@jax.jit
def compute_ising_magnetization(spins: jnp.ndarray) -> jnp.ndarray:
    """
    Compute magnetization of Ising configuration.

    M = (1/N) * sum_i s_i

    Args:
        spins: Spin configuration(s)

    Returns:
        Magnetization (scalar or array)
    """
    if spins.ndim == 1:
        return jnp.mean(spins)
    else:
        return jnp.mean(spins, axis=-1)


@jax.jit
def compute_ising_correlation(spins: jnp.ndarray) -> jnp.ndarray:
    """
    Compute spin-spin correlation matrix.

    C_ij = <s_i s_j> - <s_i><s_j>

    Args:
        spins: Spin configurations (batch, N)

    Returns:
        Correlation matrix (N, N)
    """
    mean_spins = jnp.mean(spins, axis=0, keepdims=True)
    centered = spins - mean_spins
    correlation = jnp.dot(centered.T, centered) / spins.shape[0]
    return correlation