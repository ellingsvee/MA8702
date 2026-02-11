"""Black-Box Variational Inference (BBVI) with the reparameterisation trick.

Variational family: diagonal Gaussian  q(z; mu, sigma) = N(z | mu, diag(sigma^2)).
Optimiser: Adam (via optax).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array


class BBVIResult(NamedTuple):
    mu: Array
    sigma: Array
    elbos: Array


def run_bbvi(
    key: Array,
    log_joint_fn: Callable[[Array], float],
    dim: int,
    *,
    n_iter: int = 2000,
    n_samples: int = 8,
    lr: float = 0.01,
    init_sigma: float = 1.0,
) -> BBVIResult:
    """Run BBVI with the reparameterisation trick.

    Parameters
    ----------
    key : JAX PRNG key.
    log_joint_fn : Function ``z -> log p(x, z)`` (data are baked in).
    dim : Dimensionality of the latent space.
    n_iter : Number of optimisation steps.
    n_samples : Monte-Carlo samples per gradient estimate.
    lr : Learning rate for Adam.
    init_sigma : Initial standard deviation for the variational distribution.

    Returns
    -------
    BBVIResult with final ``mu``, ``sigma``, and ELBO trace.
    """
    # Parameterise sigma via log to keep it positive.
    mu = jnp.zeros(dim)
    log_sigma = jnp.full(dim, jnp.log(init_sigma))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init((mu, log_sigma))

    @jax.jit
    def neg_elbo(mu, log_sigma, key):
        sigma = jnp.exp(log_sigma)
        eps = jax.random.normal(key, shape=(n_samples, dim))
        z = mu + sigma * eps  # reparameterisation trick
        log_q = jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi) - log_sigma - 0.5 * eps**2,
            axis=1,
        )
        log_p = jax.vmap(log_joint_fn)(z)
        return -jnp.mean(log_p - log_q)

    @jax.jit
    def step(mu, log_sigma, opt_state, key):
        loss, grads = jax.value_and_grad(neg_elbo, argnums=(0, 1))(mu, log_sigma, key)
        updates, opt_state_new = optimizer.update(grads, opt_state, (mu, log_sigma))
        mu_new, log_sigma_new = optax.apply_updates((mu, log_sigma), updates)
        return mu_new, log_sigma_new, opt_state_new, loss

    elbos = []
    for i in range(n_iter):
        key, subkey = jax.random.split(key)
        mu, log_sigma, opt_state, loss = step(mu, log_sigma, opt_state, subkey)
        elbos.append(-float(loss))

    return BBVIResult(mu=mu, sigma=jnp.exp(log_sigma), elbos=jnp.array(elbos))
