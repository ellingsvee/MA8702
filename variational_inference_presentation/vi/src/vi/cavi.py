"""Coordinate Ascent Variational Inference (CAVI) for a Bayesian Gaussian Mixture.

Model (D-dimensional observations):
    mu_k ~ N(0, sigma^2 I_D)           for k = 1, ..., K
    c_i  ~ Categorical(1/K)            for i = 1, ..., N
    x_i | c_i, mu ~ N(mu_{c_i}, I_D)   for i = 1, ..., N   (unit covariance)

Mean-field factorisation:
    q(mu, c) = [prod_k N(mu_k | m_k, s_k^2 I_D)] * [prod_i Cat(c_i | phi_i)]

Shapes:
    x:   (N, D)   observations
    m:   (K, D)   posterior means for mu_k
    s2:  (K,)     posterior variance (isotropic scalar per component)
    phi: (N, K)   assignment probabilities
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class GMMPrior(NamedTuple):
    sigma: float = 5.0  # prior std for each mu_k
    K: int = 3  # number of mixture components


class CAVIPosterior(NamedTuple):
    m: Array  # shape (K, D) — posterior means for mu_k
    s2: Array  # shape (K,) — posterior variances (isotropic)
    phi: Array  # shape (N, K) — assignment probabilities


def _compute_elbo(
    x: Array,
    prior: GMMPrior,
    post: CAVIPosterior,
) -> Array:
    """Compute the evidence lower bound (ELBO) for D-dimensional GMM."""
    K = prior.K
    sigma2 = prior.sigma**2
    m, s2, phi = post.m, post.s2, post.phi
    N, D = x.shape

    # E_q[log p(x | c, mu)]
    # = sum_ik phi_{ik} [-D/2 log(2pi) - 0.5(||x_i||^2 - 2 x_i^T m_k + ||m_k||^2 + D s_k^2)]
    x_sq = jnp.sum(x**2, axis=1)  # (N,)
    m_sq = jnp.sum(m**2, axis=1)  # (K,)
    xm = x @ m.T  # (N, K)
    sq_term = x_sq[:, None] - 2 * xm + m_sq[None, :] + D * s2[None, :]  # (N, K)
    log_lik = jnp.sum(phi * (-0.5 * D * jnp.log(2 * jnp.pi) - 0.5 * sq_term))

    # E_q[log p(c)] = N log(1/K)
    log_p_c = N * jnp.log(1.0 / K)

    # E_q[log p(mu)] = sum_k [-D/2 log(2pi sigma^2) - (||m_k||^2 + D s_k^2)/(2 sigma^2)]
    log_p_mu = jnp.sum(
        -0.5 * D * jnp.log(2 * jnp.pi * sigma2)
        - (m_sq + D * s2) / (2 * sigma2)
    )

    # H[q(mu)] = sum_k D/2 (1 + log(2 pi s_k^2))
    H_mu = jnp.sum(0.5 * D * (1 + jnp.log(2 * jnp.pi * s2)))

    # H[q(c)] = -sum_ik phi_{ik} log(phi_{ik})
    H_c = -jnp.sum(phi * jnp.log(phi + 1e-40))

    return log_lik + log_p_c + log_p_mu + H_mu + H_c


def run_cavi(
    x: Array,
    prior: GMMPrior = GMMPrior(),
    n_iter: int = 100,
) -> tuple[CAVIPosterior, Array]:
    """Run CAVI and return the posterior and per-iteration ELBO trace.

    Parameters
    ----------
    x : Array of observations, shape (N,) or (N, D).
    prior : Hyper-parameters of the GMM prior.
    n_iter : Number of coordinate-ascent sweeps.

    Returns
    -------
    posterior : CAVIPosterior with final variational parameters.
    elbos : Array of ELBO values (length ``n_iter``).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim == 1:
        x = x[:, None]  # (N,) -> (N, 1)
    N, D = x.shape
    K = prior.K
    sigma2 = prior.sigma**2

    # Initialise phi uniformly
    phi = jnp.ones((N, K)) / K

    # Initialise m by picking K random data points spread across the data
    indices = jnp.linspace(0, N - 1, K).astype(int)
    m = x[indices]  # (K, D)
    s2 = jnp.ones(K)

    elbos = []

    for _ in range(n_iter):
        # --- update q(c_i): phi_{ik} propto exp(x_i^T m_k - 0.5(||m_k||^2 + D s_k^2)) ---
        m_sq = jnp.sum(m**2, axis=1)  # (K,)
        log_phi = x @ m.T - 0.5 * (m_sq[None, :] + D * s2[None, :])  # (N, K)
        log_phi = log_phi - jnp.max(log_phi, axis=1, keepdims=True)
        phi = jnp.exp(log_phi)
        phi = phi / jnp.sum(phi, axis=1, keepdims=True)

        # --- update q(mu_k) ---
        # s_k^2 = 1/(1/sigma^2 + N_k)
        # m_k   = s_k^2 * sum_i phi_{ik} x_i   (D-dim vector)
        N_k = jnp.sum(phi, axis=0)  # (K,)
        s2 = 1.0 / (1.0 / sigma2 + N_k)  # (K,)
        m = s2[:, None] * (phi.T @ x)  # (K, D)

        post = CAVIPosterior(m=m, s2=s2, phi=phi)
        elbos.append(float(_compute_elbo(x, prior, post)))

    return post, jnp.array(elbos)
