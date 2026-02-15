import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple


class MultivariateADVIResult(NamedTuple):
    """Result of ADVI for multivariate linear regression.

    Mean-field Gaussian in unconstrained space:
        q(zeta) = N(mu_full, diag(sigma_full^2))

    where zeta = (beta_1, ..., beta_p, log(sigma^2)).
    """

    mu: jnp.ndarray  # (p,)   posterior mean of beta
    Sigma: jnp.ndarray  # (p,p)  posterior covariance of beta (diagonal)
    mu_full: jnp.ndarray  # (p+1,) full variational mean incl. log_sigma2
    sigma_full: jnp.ndarray  # (p+1,) full variational std devs
    elbo_history: jnp.ndarray  # (n_steps,) ELBO trace


@jax.jit(static_argnames=("logdensity_fn", "d", "n_steps", "n_samples"))
def advi_multivariate(
    logdensity_fn,
    d,
    *,
    n_steps=5000,
    n_samples=8,
    learning_rate=0.01,
    key,
):
    """Automatic Differentiation Variational Inference (mean-field Gaussian).

    Parameters
    ----------
    logdensity_fn : callable
        Log-density of the target distribution.  Signature: (params,) -> scalar.
    d : int
        Dimensionality of the unconstrained parameter vector.
    n_steps : int
        Number of optimisation steps.
    n_samples : int
        Number of Monte Carlo samples per ELBO estimate.
    learning_rate : float
        Adam learning rate.
    key : jax.random.PRNGKey
        Random key.
    """
    optimizer = optax.adam(learning_rate)

    # Variational parameters: mu (d,) and omega (d,) where sigma = exp(omega)
    mu = jnp.zeros(d)
    omega = jnp.full(d, -1.0)  # start with small std devs
    opt_state = optimizer.init((mu, omega))

    log_density_vmap = jax.vmap(logdensity_fn)

    def neg_elbo(mu, omega, key):
        eps = jax.random.normal(key, shape=(n_samples, d))
        sigma = jnp.exp(omega)
        zeta = mu + sigma * eps  # (n_samples, d) reparameterisation trick

        # E_q[log p(zeta)]
        log_p = log_density_vmap(zeta).mean()

        # Entropy of mean-field Gaussian: sum(omega) + d/2 * log(2*pi*e)
        entropy = jnp.sum(omega) + d / 2.0 * jnp.log(2.0 * jnp.pi * jnp.e)

        return -(log_p + entropy)

    grad_fn = jax.grad(neg_elbo, argnums=(0, 1))

    def step(carry, _):
        mu, omega, opt_state, key = carry
        key, subkey = jax.random.split(key)

        grads = grad_fn(mu, omega, subkey)
        updates, opt_state_new = optimizer.update(grads, opt_state, (mu, omega))
        mu_new, omega_new = optax.apply_updates((mu, omega), updates)

        elbo = -neg_elbo(mu_new, omega_new, subkey)
        return (mu_new, omega_new, opt_state_new, key), elbo

    init_carry = (mu, omega, opt_state, key)
    (mu_final, omega_final, _, _), elbo_history = jax.lax.scan(step, init_carry, None, length=n_steps)

    sigma_final = jnp.exp(omega_final)

    # Extract beta portion (first d-1 components)
    p = d - 1
    mu_beta = mu_final[:p]
    sigma_beta = sigma_final[:p]
    Sigma_beta = jnp.diag(sigma_beta**2)

    return MultivariateADVIResult(
        mu=mu_beta,
        Sigma=Sigma_beta,
        mu_full=mu_final,
        sigma_full=sigma_final,
        elbo_history=elbo_history,
    )
