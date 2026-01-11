import jax
from jax import Array
import jax.numpy as jnp
from typing import NamedTuple, Callable


class LangevinState(NamedTuple):
    position: Array
    logdensity: Array
    logdensity_grad: Array

class LangevinInfo(NamedTuple):
    acceptance_rate: Array
    is_accepted: Array
    proposal: LangevinState

def init(position: Array, logdensity_fn: Callable) -> LangevinState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return LangevinState(position, logdensity, logdensity_grad)

def build_kernel(logdensity_fn: Callable, step_size: float) -> Callable:

    def kernel(
        rng_key: Array, state: LangevinState,
    ) -> tuple[LangevinState, LangevinInfo]:
        # Generate proposal: x' = x + step_size * N(0, I)
        key_proposal, key_accept = jax.random.split(rng_key)

        proposal = state.position + step_size * state.logdensity_grad + jnp.sqrt(2 * step_size) * jax.random.normal(
            key_proposal, shape=state.position.shape
        )

        proposal_logdensity, proposal_logdensity_grad = jax.value_and_grad(logdensity_fn)(proposal)

        # Compute acceptance ratio (symmetric proposal cancels out)
        log_ratio = proposal_logdensity - state.logdensity
        # Compute the proposal densities q(x'|x) and q(x|x')
        def log_proposal_density(from_pos, to_pos, from_grad):
            diff = to_pos - from_pos - step_size * from_grad
            return -0.5 * jnp.sum(diff ** 2) / (2 * step_size)
        log_q_forward = log_proposal_density(state.position, proposal, state.logdensity_grad)
        log_q_backward = log_proposal_density(proposal, state.position, proposal_logdensity_grad)
        log_ratio += log_q_backward - log_q_forward
        acceptance_prob = jnp.minimum(1.0, jnp.exp(log_ratio))

        # Accept or reject
        u = jax.random.uniform(key_accept)
        accepted = u < acceptance_prob

        # Update state (use lax.cond for cleaner scalar handling)
        new_position = jax.lax.select(accepted, proposal, state.position)
        new_logdensity = jax.lax.select(accepted, proposal_logdensity, state.logdensity)
        new_logdensity_grad = jax.lax.select(accepted, proposal_logdensity_grad, state.logdensity_grad)
        new_state = LangevinState(new_position, new_logdensity, new_logdensity_grad)

        # Store info
        info = LangevinInfo(acceptance_prob, accepted, LangevinState(proposal, proposal_logdensity, proposal_logdensity_grad))

        return new_state, info

    return kernel

