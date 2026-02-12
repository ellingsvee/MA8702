import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.nn import logsumexp

from vi.gmm import GMMPrior


def run_cavi(
    x: Array,
    gmm_prior: GMMPrior = GMMPrior(),
    n_iter: int = 100,
):
    if x.ndim == 1:
        x = x[:, None]
    n, dim = x.shape
    K = gmm_prior.K

    sigma2 = gmm_prior.sigma**2

    # Initialize variational parameters
    indices = jnp.linspace(0, n - 1, K).astype(int)
    m = x[indices]
    s2 = jnp.ones((K,))
    phi = jnp.ones((n, K)) / K

    parameters = (m, s2, phi)

    @jit
    def step(parameters):
        m, s2, phi = parameters

        linear = x @ m.T
        quad = jnp.sum(m**2, axis=1) + dim * s2

        log_phi = linear - 0.5 * quad
        log_phi = log_phi - logsumexp(log_phi, axis=1, keepdims=True)
        phi = jnp.exp(log_phi)

        Nk = jnp.sum(phi, axis=0)
        x_weighted = phi.T @ x

        # Posterior variance:
        s2 = 1.0 / (1.0 / sigma2 + Nk)

        # Posterior mean:
        m = s2[:, None] * x_weighted

        return m, s2, phi

    for _ in range(n_iter):
        parameters = step(parameters)

    return parameters


def sample_cavi(key, parameters, n_samples=8):
    m, s2, phi = parameters
    K, dim = m.shape

    key, subkey = jax.random.split(key)
    component_samples = jax.random.categorical(subkey, jnp.log(phi), shape=(n_samples,))

    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=(n_samples, dim))
    samples = m[component_samples] + jnp.sqrt(s2[component_samples]) * eps
    return samples
