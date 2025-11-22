"""
Sampling algorithms for thermodynamic models
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple, Callable
from .state import PBitState, init_pbit_state
from .energy_models import ising_energy


@partial(jax.jit, static_argnames=["num_steps", "record_trajectory"])
def gibbs_sample_ising(
    J: jnp.ndarray,
    h: jnp.ndarray,
    beta: float,
    num_steps: int,
    key: jax.random.PRNGKey,
    init_state: Optional[jnp.ndarray] = None,
    record_trajectory: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    """
    Gibbs sampling for Ising model.

    Args:
        J: Coupling matrix (N, N)
        h: External field (N,)
        beta: Inverse temperature
        num_steps: Number of Gibbs steps
        key: Random key
        init_state: Initial spin configuration
        record_trajectory: Whether to record full trajectory

    Returns:
        Final state and dictionary with additional info
    """
    n_spins = J.shape[0]

    # Initialize
    if init_state is None:
        key, subkey = jax.random.split(key)
        state = jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_spins,))
    else:
        state = init_state

    def gibbs_step(carry, _):
        state, key, accepted = carry
        key, key_site, key_flip = jax.random.split(key, 3)

        # Random site selection
        site = jax.random.randint(key_site, (), 0, n_spins)

        # Compute local field
        local_field = jnp.dot(J[site], state) + h[site]

        # Probability of spin up
        prob_up = jax.nn.sigmoid(2 * beta * local_field)

        # Sample new spin
        new_spin = jax.lax.cond(
            jax.random.uniform(key_flip) < prob_up,
            lambda _: 1.0,
            lambda _: -1.0,
            None,
        )

        # Update state
        state = state.at[site].set(new_spin)

        # Track acceptance (Gibbs always accepts)
        accepted = accepted + 1

        return (state, key, accepted), state

    # Run sampling
    init_carry = (state, key, 0)
    (final_state, final_key, total_accepted), trajectory = jax.lax.scan(
        gibbs_step, init_carry, jnp.arange(num_steps)
    )

    # Compute final energy
    final_energy = ising_energy(final_state, J, h)

    info = {
        "energy": final_energy,
        "acceptance_rate": total_accepted / num_steps,
        "trajectory": trajectory if record_trajectory else None,
    }

    return final_state, info


@partial(jax.jit, static_argnames=["num_steps"])
def metropolis_sample_ising(
    J: jnp.ndarray,
    h: jnp.ndarray,
    beta: float,
    num_steps: int,
    key: jax.random.PRNGKey,
    init_state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, dict]:
    """
    Metropolis-Hastings sampling for Ising model.

    Args:
        J: Coupling matrix (N, N)
        h: External field (N,)
        beta: Inverse temperature
        num_steps: Number of MH steps
        key: Random key
        init_state: Initial spin configuration

    Returns:
        Final state and info dictionary
    """
    n_spins = J.shape[0]

    # Initialize
    if init_state is None:
        key, subkey = jax.random.split(key)
        state = jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_spins,))
    else:
        state = init_state

    current_energy = ising_energy(state, J, h)

    def metropolis_step(carry, _):
        state, energy, key, accepted = carry
        key, key_site, key_accept = jax.random.split(key, 3)

        # Propose flip
        site = jax.random.randint(key_site, (), 0, n_spins)
        proposed_state = state.at[site].multiply(-1)

        # Compute energy change
        proposed_energy = ising_energy(proposed_state, J, h)
        delta_energy = proposed_energy - energy

        # Metropolis acceptance
        accept_prob = jnp.minimum(1.0, jnp.exp(-beta * delta_energy))
        accept = jax.random.uniform(key_accept) < accept_prob

        # Update state conditionally
        new_state = jax.lax.cond(
            accept,
            lambda _: proposed_state,
            lambda _: state,
            None,
        )
        new_energy = jax.lax.cond(
            accept,
            lambda _: proposed_energy,
            lambda _: energy,
            None,
        )
        new_accepted = accepted + accept.astype(jnp.float32)

        return (new_state, new_energy, key, new_accepted), new_state

    # Run sampling
    init_carry = (state, current_energy, key, 0.0)
    (final_state, final_energy, _, total_accepted), trajectory = jax.lax.scan(
        metropolis_step, init_carry, jnp.arange(num_steps)
    )

    info = {
        "energy": final_energy,
        "acceptance_rate": total_accepted / num_steps,
    }

    return final_state, info


@partial(jax.jit, static_argnames=["num_steps"])
def parallel_tempering_ising(
    J: jnp.ndarray,
    h: jnp.ndarray,
    betas: jnp.ndarray,
    num_steps: int,
    key: jax.random.PRNGKey,
    swap_interval: int = 10,
) -> Tuple[jnp.ndarray, dict]:
    """
    Parallel tempering (replica exchange) for Ising model.

    Args:
        J: Coupling matrix (N, N)
        h: External field (N,)
        betas: Array of inverse temperatures (n_replicas,)
        num_steps: Number of sampling steps
        key: Random key
        swap_interval: Steps between replica swaps

    Returns:
        States from all temperatures and info
    """
    n_spins = J.shape[0]
    n_replicas = betas.shape[0]

    # Initialize replicas
    key, *subkeys = jax.random.split(key, n_replicas + 1)
    states = jax.vmap(
        lambda k: jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(n_spins,))
    )(jnp.array(subkeys))

    energies = jax.vmap(lambda s: ising_energy(s, J, h))(states)

    def replica_step(carry, step_idx):
        states, energies, key = carry

        # Gibbs updates for each replica
        def update_replica(state, beta, k):
            new_state, info = gibbs_sample_ising(
                J, h, beta, 1, k, state, False
            )
            return new_state, info["energy"]

        keys = jax.random.split(key, n_replicas + 1)
        key = keys[0]

        new_states, new_energies = jax.vmap(update_replica)(
            states, betas, keys[1:]
        )

        # Attempt replica swaps
        def maybe_swap(states, energies, key):
            key, key_pair, key_accept = jax.random.split(key, 3)

            # Random pair of adjacent replicas
            pair_idx = jax.random.randint(key_pair, (), 0, n_replicas - 1)

            # Compute swap probability
            beta_diff = betas[pair_idx + 1] - betas[pair_idx]
            energy_diff = energies[pair_idx + 1] - energies[pair_idx]
            swap_prob = jnp.minimum(1.0, jnp.exp(beta_diff * energy_diff))

            # Perform swap conditionally
            swap = jax.random.uniform(key_accept) < swap_prob

            def do_swap(s, e):
                s = s.at[pair_idx].set(states[pair_idx + 1])
                s = s.at[pair_idx + 1].set(states[pair_idx])
                e = e.at[pair_idx].set(energies[pair_idx + 1])
                e = e.at[pair_idx + 1].set(energies[pair_idx])
                return s, e

            return jax.lax.cond(
                swap,
                lambda _: do_swap(states, energies),
                lambda _: (states, energies),
                None,
            )

        # Conditionally perform swaps
        states, energies = jax.lax.cond(
            step_idx % swap_interval == 0,
            lambda _: maybe_swap(new_states, new_energies, key),
            lambda _: (new_states, new_energies),
            None,
        )

        return (states, energies, key), states

    # Run parallel tempering
    init_carry = (states, energies, key)
    (final_states, final_energies, _), trajectory = jax.lax.scan(
        replica_step, init_carry, jnp.arange(num_steps)
    )

    info = {
        "energies": final_energies,
        "betas": betas,
    }

    return final_states, info


@jax.jit
def langevin_sample(
    energy_fn: Callable,
    init_state: jnp.ndarray,
    beta: float,
    num_steps: int,
    step_size: float,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, dict]:
    """
    Langevin dynamics sampling for continuous states.

    Args:
        energy_fn: Energy function
        init_state: Initial state
        beta: Inverse temperature
        num_steps: Number of steps
        step_size: Step size for updates
        key: Random key

    Returns:
        Final state and info
    """

    def langevin_step(carry, _):
        state, key = carry
        key, noise_key = jax.random.split(key)

        # Compute gradient
        grad_energy = jax.grad(energy_fn)(state)

        # Langevin update
        noise = jax.random.normal(noise_key, state.shape)
        state = state - step_size * grad_energy + jnp.sqrt(2 * step_size / beta) * noise

        return (state, key), state

    # Run dynamics
    (final_state, _), trajectory = jax.lax.scan(
        langevin_step, (init_state, key), jnp.arange(num_steps)
    )

    info = {
        "final_energy": energy_fn(final_state),
    }

    return final_state, info