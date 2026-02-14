import jax
from jax import Array
import jax.numpy as jnp
from typing import NamedTuple, Callable


class RWState(NamedTuple):
    position: Array
    logdensity: Array


class RWInfo(NamedTuple):
    acceptance_rate: Array
    is_accepted: Array
    proposal: RWState


def init(position: Array, logdensity_fn: Callable) -> RWState:
    return RWState(position, logdensity_fn(position))


def build_kernel(logdensity_fn: Callable, step_size: float) -> Callable:
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: Array,
        state: RWState,
    ) -> tuple[RWState, RWInfo]:
        # Generate proposal: x' = x + step_size * N(0, I)
        key_proposal, key_accept = jax.random.split(rng_key)
        proposal = state.position + step_size * jax.random.normal(
            key_proposal, shape=state.position.shape
        )

        # Compute log probability at proposal
        proposal_logdensity = logdensity_fn(proposal)

        # Compute acceptance ratio (symmetric proposal cancels out)
        log_ratio = proposal_logdensity - state.logdensity
        acceptance_prob = jnp.minimum(1.0, jnp.exp(log_ratio))

        # Accept or reject
        u = jax.random.uniform(key_accept)
        accepted = u < acceptance_prob

        # Update state (use lax.cond for cleaner scalar handling)
        new_position = jax.lax.select(accepted, proposal, state.position)
        new_logdensity = jax.lax.select(accepted, proposal_logdensity, state.logdensity)
        new_state = RWState(new_position, new_logdensity)

        # Store info
        info = RWInfo(acceptance_prob, accepted, RWState(proposal, proposal_logdensity))

        return new_state, info

    return kernel
