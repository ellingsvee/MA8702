import jax
import jax.numpy as jnp
from tabulate import tabulate

from vi.bbvi import run_bbvi
from vi.data import generate_gmm_dataset

jax.config.update("jax_enable_x64", True)

K = 3
PRIOR_SIGMA = 5.0

x, centers = generate_gmm_dataset(K=K, n_samples=300, seed=42)
n, dim = x.shape


# log p(x, mu) with the discrete assignments c_i marginalised out.
# BBVI needs a smooth function of the latent variables, so we integrate
# out the discrete c_i analytically and only do VI over the continuous mu.
#
# log p(x, mu) = log p(x | mu) + log p(mu)
#   where log p(x | mu) = sum_i log [sum_k (1/K) N(x_i | mu_k, I)]
#   and   log p(mu)      = sum_k log N(mu_k | 0, sigma^2 I)
def log_joint(mu):
    # log p(x | mu) with c marginalised out
    diff = x[:, None, :] - mu[None, :, :]  # (n, K, dim)
    log_components = (
        -0.5 * dim * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.sum(diff**2, axis=2)
        + jnp.log(
            1.0 / K
        )  # This is because of p(c_i = k) = 1/K for all k, which we marginalise out
    )  # (n, K)
    log_lik = jnp.sum(jax.nn.logsumexp(log_components, axis=1))

    # log p(mu)
    log_prior = jnp.sum(
        -0.5 * dim * jnp.log(2 * jnp.pi * PRIOR_SIGMA**2)
        - 0.5 * jnp.sum(mu**2, axis=1) / PRIOR_SIGMA**2
    )

    return log_lik + log_prior


mu_est, sigma_est, elbos = run_bbvi(
    key=jax.random.key(123),
    log_joint_fn=log_joint,
    shape=(K, dim),
    n_iter=3000,
    n_samples=16,
    lr=0.02,
)

# Sort by first coordinate for display
order = jnp.argsort(mu_est[:, 0])
mu_sorted = mu_est[order]
sigma_sorted = sigma_est[order]
centers_sorted = centers[jnp.argsort(centers[:, 0])]

headers = (
    ["Cluster"]
    + [f"True mu[{d}]" for d in range(dim)]
    + [f"Est mu[{d}]" for d in range(dim)]
    + [f"sigma[{d}]" for d in range(dim)]
)
table = jnp.concatenate(
    [
        jnp.arange(K)[:, None],
        centers_sorted,
        mu_sorted,
        sigma_sorted,
    ],
    axis=1,
)
print(tabulate(table.tolist(), headers=headers, tablefmt="fancy_grid", floatfmt=".3f"))
