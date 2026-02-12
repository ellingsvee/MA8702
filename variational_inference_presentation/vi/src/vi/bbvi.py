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
    # Goal: approximate the posterior p(z|x) with a simple distribution q(z).
    #
    # log_joint_fn computes log p(x, z) for a given z — this encodes our model.
    # It is the only model-specific part; everything else here is generic.
    #
    # We choose q(z) = N(z | mu, diag(sigma^2)) and optimise mu, sigma
    # to maximise the ELBO = E_q[log p(x,z) - log q(z)].

    # Optimise log_sigma instead of sigma so that sigma = exp(log_sigma) > 0 always.
    mu = jnp.zeros(shape)
    log_sigma = jnp.full(shape, 0.0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init((mu, log_sigma))

    @jax.jit
    def neg_elbo(mu, log_sigma, key):
        sigma = jnp.exp(log_sigma)

        # The ELBO = E_q[log p(x,z) - log q(z)] involves an expectation over q.
        # For most models this expectation has no closed form, so we approximate
        # it with Monte Carlo: draw S samples z_s ~ q, then average.
        #
        # But to optimise mu and sigma with gradient descent, we need gradients
        # of the ELBO w.r.t. mu, sigma. The problem is that z ~ q(z; mu, sigma)
        # depends on mu and sigma, so we can't just differentiate through sampling.
        #
        # Reparameterization trick: instead of sampling z ~ N(mu, sigma^2 I),
        # sample eps ~ N(0, I) and set z = mu + sigma * eps. Now the randomness
        # is in eps (independent of mu, sigma) and we can differentiate z w.r.t.
        # mu and sigma as usual.
        eps = jax.random.normal(key, shape=(n_samples, *shape))
        z = mu + sigma * eps

        # Evaluate log p(x, z) for each sample
        log_p = jax.vmap(log_joint_fn)(z)

        # Evaluate log q(z) = log N(z | mu, diag(sigma^2)) for each sample.
        # Using eps directly: log q = sum_d [-0.5 log(2pi) - log(sigma_d) - 0.5 eps_d^2]
        # Sum over all axes except the first (sample) axis.
        latent_axes = tuple(range(1, len(shape) + 1))
        log_q = jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi) - log_sigma - 0.5 * eps**2,
            axis=latent_axes,
        )

        # ELBO ≈ (1/S) sum_s [log p(x, z_s) - log q(z_s)]
        elbo = jnp.mean(log_p - log_q)
        return -elbo

    @jax.jit
    def step(mu, log_sigma, opt_state, key):
        loss, grads = value_and_grad(neg_elbo, argnums=(0, 1))(mu, log_sigma, key)
        updates, opt_state_new = optimizer.update(grads, opt_state, (mu, log_sigma))
        mu_new, log_sigma_new = optax.apply_updates((mu, log_sigma), updates)
        return mu_new, log_sigma_new, opt_state_new, loss

    elbos = []
    for _ in range(n_iter):
        key, subkey = jax.random.split(key)
        mu, log_sigma, opt_state, loss = step(mu, log_sigma, opt_state, subkey)
        elbos.append(-float(loss))

    sigma = jnp.exp(log_sigma)
    return mu, sigma, jnp.array(elbos)
