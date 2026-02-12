import jax
import optax
import jax.numpy as jnp
from jax import Array, value_and_grad
from typing import Callable


def run_bbvi(
    key: Array,
    log_joint_fn: Callable[[Array], float],
    shape: tuple[int, ...],
    n_iter: int = 2000,
    n_samples: int = 8,
    lr: float = 0.01,
):
    mu = jnp.zeros(shape)
    log_sigma = jnp.full(shape, 0.0)
    params = (mu, log_sigma)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def neg_elbo(mu, log_sigma, key):
        sigma = jnp.exp(log_sigma)

        # Reparameterisation trick: z = mu + sigma * eps, where eps ~ N(0, I)
        eps = jax.random.normal(key, shape=(n_samples, *shape))
        z = mu + sigma * eps

        # Evaluate log p(x, z) for each sample
        log_p = jax.vmap(log_joint_fn)(z)

        latent_axes = tuple(range(1, len(shape) + 1))
        log_q = jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi) - log_sigma - 0.5 * eps**2,
            axis=latent_axes,
        )

        elbo = jnp.mean(log_p - log_q)
        return -elbo

    @jax.jit
    def step(params, opt_state, key):
        mu, log_sigma = params
        loss, grads = value_and_grad(neg_elbo, argnums=(0, 1))(mu, log_sigma, key)
        updates, opt_state_new = optimizer.update(grads, opt_state, (mu, log_sigma))
        mu_new, log_sigma_new = optax.apply_updates((mu, log_sigma), updates)
        return (mu_new, log_sigma_new), opt_state_new, loss

    for _ in range(n_iter):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)

    mu, log_sigma = params
    sigma = jnp.exp(log_sigma)
    return (mu, sigma)
