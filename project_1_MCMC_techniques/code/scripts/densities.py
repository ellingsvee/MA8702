from jax import vmap, jit
import jax.numpy as jnp

@jit
def log_mvn(x, mean, cov):
    d = x.shape[1]
    cov_inv = jnp.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * jnp.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    log_norm_const = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov)))
    return log_norm_const + exponent

def log_mvn_dist(x):
    """Log density for single sample or batch of samples.

    Args:
        x: Either a single sample of shape (2,) or a batch of shape (n, 2)

    Returns:
        Scalar log probability for single sample, or array of shape (n,) for batch
    """
    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])

    # Handle both single samples and batches
    if x.ndim == 1:
        # Single sample: shape (2,) -> (1, 2) -> scalar
        x_batched = x.reshape(1, -1)
        return log_mvn(x_batched, mean, cov)[0]
    else:
        # Batch of samples: shape (n, 2) -> (n,)
        return log_mvn(x, mean, cov)
    
@jit
def multimodal(x) :
    # Handle both single samples and batches
    if x.ndim == 1:
        # Single sample: shape (2,) -> (1, 2) -> scalar
        x = x.reshape(1, -1)

    w = jnp.ones((3,)) / 3.0
    means = jnp.array([[-1.5, -1.5], [1.5, 1.5], [-2.0, 2.0]])
    
    # All three covariance matrices with 0 on the off-diagonal elements.
    # The first two have 1.0 on the diagonal, the last has 0.8.
    covs = jnp.array([jnp.eye(2), jnp.eye(2), 0.8 * jnp.eye(2)])

    # Vectorize mvn over (mean, cov), keeping x fixed
    mvn_components = vmap(
        lambda mean, cov: jnp.exp(log_mvn(x, mean, cov)),
        in_axes=(0, 0)
    )(means, covs)   # shape: (3, num_points)

    return jnp.sum(w[:, None] * mvn_components, axis=0)[0]

    

def log_volcano(x):
    # Handle both single samples and batches
    if x.ndim == 1:
        # Single sample: shape (2,) -> (1, 2) -> scalar
        x = x.reshape(1, -1)

    xtx = jnp.sum(x**2, axis=1)
    norm_const = 1.0 / (2 * jnp.pi)

    out = jnp.log(norm_const) + jnp.log(xtx + 0.25) - 0.5 * xtx
    return out[0]