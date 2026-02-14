import jax.numpy as jnp
import jax
from jax import Array


def generate_data(
    x: Array,
    beta: float = 0.30,
    sigma2: float = 1.0,
    seed: int = 42,
) -> Array:
    key = jax.random.key(seed)
    return beta * x + jax.random.normal(key, shape=x.shape) * jnp.sqrt(sigma2)
