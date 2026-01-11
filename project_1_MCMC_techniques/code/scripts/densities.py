import jax
import jax.numpy as jnp
import jax.scipy.stats as stats


def logdensity_mvn(x):
    cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    return stats.multivariate_normal.logpdf(x, mean=jnp.zeros(2), cov=cov)


def logdensity_multimodal(x):
    w = jnp.ones((3,)) / 3.0
    means = jnp.array([[-1.5, -1.5], [1.5, 1.5], [-2.0, 2.0]])

    # All three covariance matrices with 0 on the off-diagonal elements.
    # The first two have 1.0 on the diagonal, the last has 0.8.
    covs = jnp.array([jnp.eye(2), jnp.eye(2), 0.8 * jnp.eye(2)])

    # Compute log density for each component
    log_components = jax.vmap(lambda mean, cov: stats.multivariate_normal.logpdf(x, mean=mean, cov=cov), in_axes=(0, 0))(
        means, covs
    )

    # Log-sum-exp trick: log(sum w_i * exp(log_p_i)) = logsumexp(log(w_i) + log_p_i)
    log_w = jnp.log(w)
    return jax.scipy.special.logsumexp(log_w + log_components)


def logdensity_volcano(x):
    xtx = jnp.sum(x**2)
    norm_const = 1.0 / (2 * jnp.pi)
    return jnp.log(norm_const) + jnp.log(xtx + 0.25) - 0.5 * xtx
