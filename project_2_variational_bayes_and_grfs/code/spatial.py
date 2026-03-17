from os import PathLike
from typing import Callable, Union

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax
from jax import Array, jit, value_and_grad
from tqdm import tqdm


def matern(h: Array, theta: Array) -> Array:
    return theta[0] * (1.0 + theta[1] * h) * jnp.exp(-theta[1] * h)


def mean_function(s: Array) -> Array:
    return (s[:, 0] - 0.5) + (s[:, 1] - 0.5)


def covariance_matrix(s: Array, cov_function: Callable) -> Array:
    h = cdist(s, s)
    return cov_function(h)


def sample_centered_grf(key, s: Array, cov_function: Callable) -> Array:
    n = s.shape[0]
    s64 = s.astype(jnp.float64)
    z = jr.normal(key, (n,), dtype=jnp.float64)

    jitter = 1e-6 * jnp.eye(n, dtype=jnp.float64)
    cov = covariance_matrix(s64, cov_function)
    L = jnp.linalg.cholesky(cov + jitter)
    # samples = mean_function(s64) + L @ z
    samples = L @ z
    return samples


def estimate_parameters(
    s_obs: Array,
    y_obs: Array,
    init_mean_param: Array,
    init_cov_params: Array,
    init_obs_noise_param: Array,
    cov_function: Callable,
    mean_function: Callable,
    max_iter: int = 10_000,
) -> tuple[tuple[Array, Array, Array], Array]:
    # Evaluate the mean function at the observed locations
    X = mean_function(s_obs)

    # Distance matrix
    H = cdist(s_obs, s_obs)  # shape (n_obs, n_obs)

    n = y_obs.shape[0]

    def neg_log_likelihood(raw_theta: Array) -> Array:
        mean_param = raw_theta[0]
        cov_params = jnp.exp(raw_theta[1:-1])
        obs_noise_param = jnp.exp(raw_theta[-1])

        C = cov_function(H, cov_params) + obs_noise_param**2 * jnp.eye(n)
        L = jnp.linalg.cholesky(C)

        Z = y_obs - X * mean_param

        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        alpha = jsp.linalg.cho_solve((L, True), Z)
        quad = Z @ alpha

        return 0.5 * (logdet + quad + n * jnp.log(2.0 * jnp.pi))

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(neg_log_likelihood)(params)
        # print(f"Loss: {loss:.4f}, Grads: {grads:.4f}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=max_iter)
    # optimizer = optax.adam(schedule)
    optimizer = optax.adam(0.01)
    params = jnp.concatenate(
        [init_mean_param, jnp.log(init_cov_params), jnp.log(init_obs_noise_param)]
    )
    opt_state = optimizer.init(params)

    lls = []

    pbar = tqdm(range(max_iter), desc="Estimating parameters")
    for i in pbar:
        params, opt_state, loss = step(params, opt_state)
        lls.append(-loss)
        pbar.set_postfix(loss=f"{loss:.4f}")

    return (params[0], jnp.exp(params[1:-1]), jnp.exp(params[-1])), jnp.array(lls)


def kriging(
    s_pred: Array,
    s_obs: Array,
    y_obs: Array,
    mean_param: Array,
    cov_params: Array,
    obs_noise_param: Array,
    cov_function: Callable,
    mean_function: Callable,
) -> tuple[Array, Array]:
    n_obs = s_obs.shape[0]
    n_pred = s_pred.shape[0]

    X_obs = mean_function(s_obs)
    X_pred = mean_function(s_pred)

    H_obs = cdist(s_obs, s_obs)
    C = cov_function(H_obs, cov_params) + obs_noise_param**2 * jnp.eye(n_obs)
    Cinv = jnp.linalg.inv(C)

    H_pred = cdist(s_pred, s_obs)
    C_pred = cov_function(H_pred, cov_params)
    # + obs_noise_param**2 * jnp.eye(n_pred)

    mean_pred = X_pred * mean_param + C_pred @ Cinv @ (y_obs - X_obs * mean_param)
    cov_pred = obs_noise_param**2 * jnp.eye(n_pred) + C_pred @ Cinv @ C_pred.T
    var_pred = jnp.diag(cov_pred)

    return mean_pred, var_pred


@jit
def cdist(x: Array, y: Array) -> Array:
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


def generate_grid(N: int = 100) -> Array:
    x = jnp.linspace(0, 1, N)
    return jnp.stack(jnp.meshgrid(x, x), -1).reshape(-1, 2)


def observation_matrix(n: int, obs_indices: Array) -> Array:
    """
    Projection matrix of shape (N_obs, n) that extracts observed locations.
    """
    N_obs = obs_indices.shape[0]
    return jnp.zeros((N_obs, n)).at[jnp.arange(N_obs), obs_indices].set(1.0)


def random_observation_matrix(key, s: Array, N_obs: int = 100) -> tuple[Array, Array]:
    """
    Randomly select `N_obs` observation points from `s` and return the corresponding observation matrix.
    """
    n = s.shape[0]
    obs_indices = jr.choice(key, n, (N_obs,), replace=False)
    obs_points = s[obs_indices]
    return obs_points, observation_matrix(n, obs_indices)


def cross_mask(s: Array, arm_width: float = 0.15) -> Array:
    """Boolean mask selecting grid points that lie in a cross/plus shape centered at (0.5, 0.5)."""
    horizontal = (
        (jnp.abs(s[:, 1] - 0.5) < arm_width / 2) & (s[:, 0] > 0.1) & (s[:, 0] < 0.9)
    )
    vertical = (
        (jnp.abs(s[:, 0] - 0.5) < arm_width / 2) & (s[:, 1] > 0.1) & (s[:, 1] < 0.9)
    )
    return horizontal | vertical


def plot_sample(
    s: Array,
    sample: Array,
    filename: Union[PathLike, None] = None,
    heatmap: bool = False,
    cmap: str = "viridis",
    vmin: Union[float, None] = -3,
    vmax: Union[float, None] = 3,
):
    if heatmap:
        n = int(jnp.sqrt(len(sample)))
        plt.imshow(
            sample.reshape(n, n),
            origin="lower",
            extent=[s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        plt.scatter(s[:, 0], s[:, 1], c=sample, cmap="viridis", vmin=vmin, vmax=vmax)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar()

    plt.gca().set_aspect("equal", adjustable="box")

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_ll_history(
    loglikelihood_history: Array, filename: Union[PathLike, None] = None
):
    plt.plot(loglikelihood_history)
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")

    # Adjust figsize and layout
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
