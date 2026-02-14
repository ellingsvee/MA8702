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


def generate_multivariate_data(
    key: Array,
    X: Array,
    beta: Array,
    sigma2: float = 1.0,
) -> Array:
    """Generate y = X @ beta + epsilon, epsilon ~ N(0, sigma^2 * I_n)."""
    n = X.shape[0]
    return X @ beta + jax.random.normal(key, shape=(n,)) * jnp.sqrt(sigma2)


def make_multivariate_logdensity(X, y, tau2):
    """Log-posterior for multivariate regression, parameterised as (beta, log(sigma^2)).

    params is a vector of length p+1: [beta_1, ..., beta_p, log(sigma^2)].
    """
    n, p = X.shape

    def logdensity_fn(params):
        beta = params[:p]
        log_sigma2 = params[p]
        sigma2 = jnp.exp(log_sigma2)

        # Likelihood: -n/2 log(2pi sigma^2) - 1/(2 sigma^2) ||y - X beta||^2
        residual = y - X @ beta
        ll = -n / 2 * jnp.log(2 * jnp.pi * sigma2) - 0.5 / sigma2 * (residual @ residual)

        # Prior on beta: N(0, tau^2 * sigma^2 * I_p)
        lp_beta = -p / 2 * jnp.log(2 * jnp.pi * tau2 * sigma2)
        lp_beta -= 0.5 / (tau2 * sigma2) * (beta @ beta)

        # Jeffrey's prior on sigma^2: p(sigma^2) proportional to 1/sigma^2
        # => log p = -log(sigma^2) = -log_sigma2
        lp_sigma2 = -log_sigma2

        return ll + lp_beta + lp_sigma2

    return logdensity_fn
