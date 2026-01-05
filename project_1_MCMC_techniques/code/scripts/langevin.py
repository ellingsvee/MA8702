"""
Langevin Metropolis-Hastings MCMC sampler implementation.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, NamedTuple


class LangevinState(NamedTuple):
    """State of the Langevin sampler.
    Attributes:
        position: Current position in parameter space
        log_prob: Log probability at current position
        log_prob_grad: Gradient of log probability at current position
    """

    position: Array
    log_prob: Array
    log_prob_grad: Array


class LangevinInfo(NamedTuple):
    """Information about a Langevin step.
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


def init(position: jnp.ndarray, logdensity_fn: Callable) -> LangevinState:
    """Initialize the Langevin state.

    Args:
        position: Initial position in parameter space
        logdensity_fn: Function that computes log probability

    Returns:
        Initial LangevinState
    """
    grad_fn = jax.value_and_grad(logdensity_fn)
    log_prob, log_prob_grad = grad_fn(position)
    return LangevinState(position, log_prob, log_prob_grad)


def build_kernel(logdensity_fn: Callable, step_size: float):
    """Build a Langevin kernel with fixed step size.

    Args:
        logdensity_fn: Function that computes log probability
        step_size: Standard deviation of the Gaussian proposal

    Returns:
        A kernel function that performs one Langevin step
    """

    def compute_acceptance_prob(
        position: jnp.ndarray,
        log_prob: jnp.ndarray,
        log_prob_grad: jnp.ndarray,
        proposal: jnp.ndarray,
        proposal_log_prob: jnp.ndarray,
        proposal_log_prob_grad: jnp.ndarray,
    ) -> jax.Array:
        def q(x_from, x_to, grad_from):
            diff = x_to - x_from - step_size * grad_from
            exponent = -jnp.sum(diff**2) / (4 * step_size)
            return jnp.exp(exponent)

        fraction = (
            q(proposal, position, proposal_log_prob_grad) * jnp.exp(proposal_log_prob)
        ) / (q(position, proposal, log_prob_grad) * jnp.exp(log_prob))

        return jnp.minimum(1.0, fraction)

    @jax.jit
    def kernel(key: Array, state: LangevinState) -> tuple[LangevinState, LangevinInfo]:
        """Perform one step of Langevin.

        Args:
            rng_key: JAX random key
            state: Current Langevin state

        Returns:
            Tuple of (new_state, info)
        """
        # Generate proposal: x' = x + step_size * N(0, I)
        key_proposal, key_accept = jax.random.split(key)
        noise = jax.random.normal(key_proposal, shape=state.position.shape)
        proposal = (
            state.position
            + step_size * state.log_prob_grad
            + jnp.sqrt(2 * step_size) * noise
        )

        # Compute log probability at proposal
        grad_fn = jax.value_and_grad(logdensity_fn)
        proposal_log_prob, proposal_log_prob_grad = grad_fn(proposal)

        # Compute acceptance ratio (symmetric proposal cancels out)
        acceptance_prob = compute_acceptance_prob(
            state.position,
            state.log_prob,
            state.log_prob_grad,
            proposal,
            proposal_log_prob,
            proposal_log_prob_grad,
        )

        # Accept or reject
        u = jax.random.uniform(key_accept)
        accepted = u < acceptance_prob

        # Update state (use lax.cond for cleaner scalar handling)
        new_position = jax.lax.select(accepted, proposal, state.position)
        new_log_prob = jax.lax.select(accepted, proposal_log_prob, state.log_prob)
        new_log_prob_grad = jax.lax.select(accepted, proposal_log_prob_grad, state.log_prob_grad)
        new_state = LangevinState(new_position, new_log_prob, new_log_prob_grad)

        # Store info
        info = LangevinInfo(accepted, acceptance_prob, proposal, proposal_log_prob)

        return new_state, info

    return kernel

if __name__ == "__main__":
    from time import time
    from densities import log_mvn_dist
    from mcmc import run_chain

    # Run with different sigma values
    sigma = 0.5
    num_steps = 10000
    burnin = 1000

    key = jax.random.key(42)
    initial_pos = jnp.array([0.0, 0.0])


    initial_state = init(initial_pos, log_mvn_dist)
    kernel = build_kernel(log_mvn_dist, sigma)

    # Run chain
    key, subkey = jax.random.split(key)

    start_time = time()
    samples, _, accept = run_chain(
        subkey, kernel, initial_state, num_steps
    )
    end_time = time()
    # Remove burnin
    samples = samples[burnin:]
    accept = accept[burnin:]

    # Acceptance rate
    acc_rate = float(jnp.mean(accept))

    print(f"RWMH sampling took {end_time - start_time:.3f} seconds.")
    print(f"Acceptance rate: {acc_rate:.3f}")