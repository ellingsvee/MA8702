import jax
from jax import vmap, jit
import jax.numpy as jnp

@jit
def log_mvn(x, mean, cov):
    """Log density of multivariate normal distribution.

    Args:
        x: Input point of shape (d,) - single point
        mean: Mean vector of shape (d,)
        cov: Covariance matrix of shape (d, d)

    Returns:
        Log density value (scalar)
    """
    d = x.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * jnp.dot(diff, jnp.dot(cov_inv, diff))
    log_norm_const = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov)))
    return log_norm_const + exponent

@jit
def log_mvn_dist(x):
    """Log density of the correlated bivariate normal distribution.

    Args:
        x: Input point of shape (2,) - single point

    Returns:
        Log density value (scalar)
    """
    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])
    return log_mvn(x, mean, cov)

@jit
def log_multimodal(x):
    """Log density of a mixture of three Gaussian components.

    Uses the log-sum-exp trick for numerical stability:
    log(sum_i w_i * p_i(x)) = logsumexp(log(w_i) + log(p_i(x)))

    Args:
        x: Input point of shape (2,) - single point

    Returns:
        Log density value (scalar)
    """
    w = jnp.ones((3,)) / 3.0
    means = jnp.array([[-1.5, -1.5], [1.5, 1.5], [-2.0, 2.0]])

    # All three covariance matrices with 0 on the off-diagonal elements.
    # The first two have 1.0 on the diagonal, the last has 0.8.
    covs = jnp.array([jnp.eye(2), jnp.eye(2), 0.8 * jnp.eye(2)])

    # Compute log density for each component
    log_mvn_components = vmap(
        lambda mean, cov: log_mvn(x, mean, cov),
        in_axes=(0, 0)
    )(means, covs)  # shape: (3,)

    # Log-sum-exp trick: log(sum w_i * exp(log_p_i)) = logsumexp(log(w_i) + log_p_i)
    log_w = jnp.log(w)
    return jax.scipy.special.logsumexp(log_w + log_mvn_components)

@jit
def log_volcano(x):
    """Log density of the volcano distribution.

    Args:
        x: Input point of shape (2,) - single point

    Returns:
        Log density value (scalar)
    """
    xtx = jnp.sum(x**2)
    norm_const = 1.0 / (2 * jnp.pi)
    out = jnp.log(norm_const) + jnp.log(xtx + 0.25) - 0.5 * xtx
    return out