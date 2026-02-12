import jax.numpy as jnp
import jax
from jax import Array


def generate_gmm_dataset(
    centers: Array | None = None,
    dims: int = 2,
    K: int = 3,
    n_samples: int = 300,
    seed: int = 42,
):
    key = jax.random.key(seed)

    if centers is None:
        key, key_centers = jax.random.split(key)
        centers = jax.random.normal(key_centers, shape=(K, dims)) * 5.0
    else:
        assert centers.shape == (K, dims)

    k_c, k_x = jax.random.split(key)
    assignments = jax.random.categorical(k_c, jnp.zeros(K), shape=(n_samples,))
    x = centers[assignments] + jax.random.normal(k_x, shape=(n_samples, dims))
    return x, centers
