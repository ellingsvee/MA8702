"""Coordinate Ascent Variational Inference (CAVI) for a Bayesian Gaussian model.

Model:
    x_i | mu, tau ~ N(mu, 1/tau)       for i = 1, ..., N
    mu  | tau     ~ N(mu_0, 1/(kappa_0 * tau))
    tau           ~ Gamma(a_0, b_0)

Mean-field factorisation:
    q(mu, tau) = N(mu | m_q, 1/lambda_q) * Gamma(tau | a_q, b_q)
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gammaln


class NormalGammaPrior(NamedTuple):
    mu_0: float = 0.0
    kappa_0: float = 1.0
    a_0: float = 1.0
    b_0: float = 1.0


class CAVIPosterior(NamedTuple):
    m_q: float = 0.0
    lambda_q: float = 1.0
    a_q: float = 1.0
    b_q: float = 1.0


def _compute_elbo(
    x: Array,
    prior: NormalGammaPrior,
    post: CAVIPosterior,
) -> float:
    """Compute the evidence lower bound (ELBO)."""
    N = x.shape[0]
    x_bar = jnp.mean(x)

    E_tau = post.a_q / post.b_q
    E_log_tau = digamma(post.a_q) - jnp.log(post.b_q)
    E_mu = post.m_q
    E_mu2 = post.m_q**2 + 1.0 / post.lambda_q

    # E_q[log p(x | mu, tau)]
    log_lik = 0.5 * N * (E_log_tau - jnp.log(2 * jnp.pi))
    log_lik -= 0.5 * E_tau * (
        jnp.sum((x - x_bar) ** 2)
        + N * (E_mu2 - 2 * E_mu * x_bar + x_bar**2)
    )

    # E_q[log p(mu | tau)]
    log_p_mu = 0.5 * (jnp.log(prior.kappa_0) + E_log_tau - jnp.log(2 * jnp.pi))
    log_p_mu -= 0.5 * prior.kappa_0 * E_tau * (
        E_mu2 - 2 * E_mu * prior.mu_0 + prior.mu_0**2
    )

    # E_q[log p(tau)]
    log_p_tau = (
        prior.a_0 * jnp.log(prior.b_0)
        - gammaln(prior.a_0)
        + (prior.a_0 - 1) * E_log_tau
        - prior.b_0 * E_tau
    )

    # H[q(mu)]  (entropy of normal)
    H_mu = 0.5 * (1 + jnp.log(2 * jnp.pi) - jnp.log(post.lambda_q))

    # H[q(tau)]  (entropy of gamma)
    H_tau = (
        post.a_q
        - jnp.log(post.b_q)
        + gammaln(post.a_q)
        + (1 - post.a_q) * digamma(post.a_q)
    )

    return log_lik + log_p_mu + log_p_tau + H_mu + H_tau


def run_cavi(
    x: Array,
    prior: NormalGammaPrior = NormalGammaPrior(),
    n_iter: int = 100,
) -> tuple[CAVIPosterior, Array]:
    """Run CAVI and return the posterior and per-iteration ELBO trace.

    Parameters
    ----------
    x : 1-D array of observations.
    prior : Hyper-parameters of the Normal-Gamma prior.
    n_iter : Number of coordinate-ascent sweeps.

    Returns
    -------
    posterior : CAVIPosterior with final variational parameters.
    elbos : Array of ELBO values (length ``n_iter``).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    N = x.shape[0]
    x_bar = jnp.mean(x)
    sum_sq = jnp.sum((x - x_bar) ** 2)

    # Initialise q(tau) expectations
    a_q = prior.a_0 + (N + 1) / 2.0
    b_q = prior.b_0  # initial guess, refined below
    E_tau = a_q / b_q

    elbos = []
    post = CAVIPosterior()  # will be overwritten on first iteration

    for _ in range(n_iter):
        # --- update q(mu) ---
        kappa_n = prior.kappa_0 + N
        m_q = (prior.kappa_0 * prior.mu_0 + N * x_bar) / kappa_n
        lambda_q = kappa_n * E_tau

        # --- update q(tau) ---
        E_mu = m_q
        E_mu2 = m_q**2 + 1.0 / lambda_q
        b_q = prior.b_0 + 0.5 * (
            sum_sq
            + N * (E_mu2 - 2 * E_mu * x_bar + x_bar**2)
            + prior.kappa_0 * (E_mu2 - 2 * E_mu * prior.mu_0 + prior.mu_0**2)
        )
        E_tau = a_q / b_q

        post = CAVIPosterior(m_q=m_q, lambda_q=lambda_q, a_q=a_q, b_q=b_q)
        elbos.append(float(_compute_elbo(x, prior, post)))

    return post, jnp.array(elbos)
