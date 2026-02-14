import jax.numpy as jnp
import jax
from jax import Array


def generate_data(
    key: Array,
    x: Array,
    beta: float = 0.30,
    sigma2: float = 1.0,
) -> Array:
    return beta * x + jax.random.normal(key, shape=x.shape) * jnp.sqrt(sigma2)
