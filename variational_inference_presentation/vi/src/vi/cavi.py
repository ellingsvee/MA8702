import jax.numpy as jnp
from jax import Array


def cavi(
    x: Array,
    y: Array,
    beta_init: float = 0.0,
    sigma2_init: float = 1.0,
    tau2: float = 0.25,
    seed: int = 42,
):
    # Some constants
    n = x.shape[0]
    sum_x2 = jnp.sum(x**2)
    sum_xy = jnp.sum(x * y)
    sum_y2 = jnp.sum(y**2)

    # The mu_beta is not updated in the loop, so we can compute it once
    mu_beta = sum_xy / (sum_x2 + 1 / tau2)

    # Utility functions
    def E_A(sigma2_beta):
        return 0.5 * (
            sum_y2
            - 2 * mu_beta * sum_xy
            + (sigma2_beta + mu_beta**2) * (sum_x2 + 1 / tau2)
        )

    def nu(E_A):
        return 0.5 * E_A

    # for _ in range(100):
    #     ...
