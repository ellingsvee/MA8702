import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.stats import norm


def generate_data(
    key: Array,
    x: Array,
    beta: float = 0.30,
    sigma2: float = 1.0,
) -> Array:
    return beta * x + jax.random.normal(key, shape=x.shape) * jnp.sqrt(sigma2)


def make_logdensity(x, y, tau2):
    def logdensity_fn(params):
        beta, log_sigma2 = params[0], params[1]
        sigma2 = jnp.exp(log_sigma2)

        # Likelihood
        ll = jnp.sum(norm.logpdf(y, loc=x * beta, scale=jnp.sqrt(sigma2)))

        # Priors
        lp_beta = norm.logpdf(beta, loc=0.0, scale=jnp.sqrt(tau2 * sigma2))
        lp_sigma2 = -log_sigma2

        return ll + lp_beta + lp_sigma2

    return logdensity_fn
