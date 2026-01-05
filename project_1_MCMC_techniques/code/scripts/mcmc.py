import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, NamedTuple


def run_chain(
    key: Array,
    kernel: Callable,
    initial_state: NamedTuple,
    num_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run a RWMH chain for a fixed number of steps.

    Args:
        rng_key: JAX random key
        initial_position: Starting position
        logdensity_fn: Log probability function
        step_size: Proposal step size
        num_steps: Number of MCMC steps

    Returns:
        Tuple of (samples, log_probs, acceptance_rates)
        - samples: Array of shape (num_steps, dim)
        - log_probs: Array of shape (num_steps,)
        - acceptance_rates: Array of shape (num_steps,)
    """
    # # Initialize
    # initial_state = kernel_init(initial_position, logdensity_fn)
    # kernel = kernel_builder(logdensity_fn, step_size)

    # Generate random keys
    keys = jax.random.split(key, num_steps)

    # Run chain using scan
    def scan_fn(state, key):
        new_state, info = kernel(key, state)
        return new_state, (new_state, info)

    final_state, (states, infos) = jax.lax.scan(scan_fn, initial_state, keys)

    # Extract samples and diagnostics
    samples = states.position
    log_probs = states.log_prob
    acceptance_rates = infos.accepted.astype(jnp.float32)

    return samples, log_probs, acceptance_rates
