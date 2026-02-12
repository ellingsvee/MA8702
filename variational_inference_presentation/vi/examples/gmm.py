import jax
import jax.numpy as jnp
from vi.cavi import run_cavi
from vi.bbvi import run_bbvi
from vi.data import generate_gmm_dataset
from vi.gmm import GMMPrior
from utils import tabulate_estimates


x, centers = generate_gmm_dataset(K=3, n_samples=300, seed=42)

K = 3
prior = GMMPrior(sigma=1.0, K=K)

cavi_params = run_cavi(x, prior, n_iter=100)


x, centers = generate_gmm_dataset(K=K, n_samples=300, seed=42)
n, dim = x.shape


def log_joint(mu):
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
        -0.5 * dim * jnp.log(2 * jnp.pi * prior.sigma**2)
        - 0.5 * jnp.sum(mu**2, axis=1) / prior.sigma**2
    )

    return log_lik + log_prior


bbvi_params = run_bbvi(
    key=jax.random.key(123),
    log_joint_fn=log_joint,
    shape=(K, dim),
    n_iter=3000,
    n_samples=16,
    lr=0.02,
)


print("CAVI estimates:")
tabulate_estimates(centers, cavi_params)

print("\nBBVI estimates:")
tabulate_estimates(centers, bbvi_params)
