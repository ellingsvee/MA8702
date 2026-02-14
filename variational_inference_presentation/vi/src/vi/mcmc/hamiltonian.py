from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class HMCState(NamedTuple):
    position: Array
    logdensity: Array
    logdensity_grad: Array


class HMCInfo(NamedTuple):
    acceptance_rate: Array
    is_accepted: Array
    proposal: HMCState


def init(position: Array, logdensity_fn: Callable) -> HMCState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return HMCState(position, logdensity, logdensity_grad)


def hamiltonian(logdensity, momentum):
    kinetic = 0.5 * jnp.sum(momentum**2)
    potential = -logdensity
    return potential + kinetic


def leapfrog(
    position: Array,
    momentum: Array,
    logdensity_fn: Callable,
    step_size: float,
    num_steps: int,
):
    def body_fn(_, state):
        x, p, logp, grad = state

        # Half step momentum
        p = p + 0.5 * step_size * grad

        # Full step position
        x = x + step_size * p

        # Refresh gradient
        logp, grad = jax.value_and_grad(logdensity_fn)(x)

        # Half step momentum
        p = p + 0.5 * step_size * grad

        return x, p, logp, grad

    logp0, grad0 = jax.value_and_grad(logdensity_fn)(position)

    position, momentum, logp, grad = jax.lax.fori_loop(
        0,
        num_steps,
        body_fn,
        (position, momentum, logp0, grad0),
    )

    return position, momentum, logp, grad


def build_kernel(
    logdensity_fn: Callable,
    step_size: float,
    num_steps: int = 10,
) -> Callable:
    def kernel(
        rng_key: Array,
        state: HMCState,
    ) -> tuple[HMCState, HMCInfo]:
        key_momentum, key_accept = jax.random.split(rng_key)

        # Sample momentum
        momentum0 = jax.random.normal(key_momentum, shape=state.position.shape)

        # Current Hamiltonian
        H = hamiltonian(state.logdensity, momentum0)

        # Propose new state via leapfrog integrator
        q_prop, p_prop, logp_prop, grad_prop = leapfrog(
            state.position,
            momentum0,
            logdensity_fn,
            step_size,
            num_steps,
        )

        # Proposed Hamiltonian
        H_prop = hamiltonian(logp_prop, p_prop)

        # Acceptance probability
        log_accept_ratio = H - H_prop
        acceptance_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))

        # Accept or reject
        u = jax.random.uniform(key_accept)
        accepted = u < acceptance_prob

        new_state = HMCState(
            position=jax.lax.select(accepted, q_prop, state.position),
            logdensity=jax.lax.select(accepted, logp_prop, state.logdensity),
            logdensity_grad=jax.lax.select(accepted, grad_prop, state.logdensity_grad),
        )

        info = HMCInfo(
            acceptance_rate=acceptance_prob,
            is_accepted=accepted,
            proposal=HMCState(q_prop, logp_prop, grad_prop),
        )

        return new_state, info

    return kernel
