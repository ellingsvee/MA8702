"""
Random Walk Metropolis-Hastings (RWMH) MCMC sampler implementation.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, NamedTuple

class RWMHState(NamedTuple):
    """State of the RWMH sampler.

    Attributes:
        position: Current position in parameter space
        log_prob: Log probability at current position
    """
    position: Array
    log_prob: Array


class RWMHInfo(NamedTuple):
    """Information about a RWMH step.

    Attributes:
        accepted: Whether the proposal was accepted
        acceptance_prob: Acceptance probability (min(1, ratio))
        proposal: Proposed position
        proposal_log_prob: Log probability at proposal
    """
    accepted: Array
    acceptance_prob: Array
    proposal: Array
    proposal_log_prob: Array


def init(position: jnp.ndarray, logdensity_fn: Callable) -> RWMHState:
    """Initialize the RWMH state.

    Args:
        position: Initial position in parameter space
        logdensity_fn: Function that computes log probability

    Returns:
        Initial RWMHState
    """
    log_prob = logdensity_fn(position)
    return RWMHState(position, log_prob)


def build_kernel(logdensity_fn: Callable, step_size: float):
    """Build a RWMH kernel with fixed step size.

    Args:
        logdensity_fn: Function that computes log probability
        step_size: Standard deviation of the Gaussian proposal

    Returns:
        A kernel function that performs one RWMH step
    """

    @jax.jit
    def kernel(key: Array, state: RWMHState) -> tuple[RWMHState, RWMHInfo]:
        """Perform one step of RWMH.

        Args:
            rng_key: JAX random key
            state: Current RWMH state

        Returns:
            Tuple of (new_state, info)
        """
        # Generate proposal: x' = x + step_size * N(0, I)
        key_proposal, key_accept = jax.random.split(key)
        proposal = state.position + step_size * jax.random.normal(
            key_proposal, shape=state.position.shape
        )

        # Compute log probability at proposal
        proposal_log_prob = logdensity_fn(proposal)

        # Compute acceptance ratio (symmetric proposal cancels out)
        log_ratio = proposal_log_prob - state.log_prob
        acceptance_prob = jnp.minimum(1.0, jnp.exp(log_ratio))

        # Accept or reject
        u = jax.random.uniform(key_accept)
        accepted = u < acceptance_prob

        # Update state (use lax.cond for cleaner scalar handling)
        new_position = jax.lax.select(accepted, proposal, state.position)
        new_log_prob = jax.lax.select(accepted, proposal_log_prob, state.log_prob)
        new_state = RWMHState(new_position, new_log_prob)

        # Store info
        info = RWMHInfo(accepted, acceptance_prob, proposal, proposal_log_prob)

        return new_state, info

    return kernel


def run_chain(
    key: Array,
    initial_position: jnp.ndarray,
    logdensity_fn: Callable,
    step_size: float,
    num_steps: int
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
    # Initialize
    initial_state = init(initial_position, logdensity_fn)
    kernel = build_kernel(logdensity_fn, step_size)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from densities import log_mvn_dist
    from utils import autocorr


    # Run with different sigma values
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    num_steps = 10000
    warmup = 1000

    key = jax.random.key(42)
    initial_pos = jnp.array([0.0, 0.0])

    fig, axes = plt.subplots(len(sigma_values), 3, figsize=(15, 3*len(sigma_values)))

    for i, sigma in enumerate(sigma_values):
        print(f"\nsigma = {sigma}")

        # Run chain
        key, subkey = jax.random.split(key)
        samples, _, accept = run_chain(subkey, initial_pos, log_mvn_dist, sigma, num_steps)

        # Remove warmup
        samples = samples[warmup:]
        accept = accept[warmup:]

        # Acceptance rate
        acc_rate = float(jnp.mean(accept))
        print(f"  Acceptance rate: {acc_rate:.3f}")

        row_title = rf"$\sigma = {sigma}$, Accept rate: {acc_rate:.3f}"

                # Trace plots
        axes[i, 0].plot(samples[:, 0], linewidth=0.5)
        axes[i, 0].set_ylabel(f'Trace x')

        axes[i, 1].plot(samples[:, 1], linewidth=0.5)
        axes[i, 1].set_ylabel(f'Trace y')
        axes[i, 1].set_title(row_title)

        # Autocorrelation
        acf = autocorr(samples[:, 0])
        axes[i, 2].bar(range(len(acf)), acf, width=1.0)
        axes[i, 2].set_ylabel('ACF')
        axes[i, 2].set_ylim([-0.2, 1.0])


    plt.tight_layout()
    plt.savefig('output/tuning_results.svg')
    plt.close()