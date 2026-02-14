import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gammaln
from functools import partial
from typing import NamedTuple


class CAVIResult(NamedTuple):
    """Result of CAVI for univariate linear regression.

    Variational distributions:
        q(beta) = N(mu_beta, sigma2_beta[-1])
        q(sigma^2) = InvGamma(alpha, nu[-1])
    """

    mu_beta: float
    sigma2_beta: Array
    nu: Array
    alpha: float
    elbo: Array


@partial(jax.jit, static_argnames=("max_iter",))
def cavi(
    x: Array,
    y: Array,
    sigma2_init: float = 1.0,
    tau2: float = 0.25,
    max_iter: int = 100,
) -> CAVIResult:
    n = x.shape[0]
    sum_x2 = jnp.sum(x**2)
    sum_xy = jnp.sum(x * y)
    sum_y2 = jnp.sum(y**2)

    alpha = (n + 1) / 2
    mu_beta = sum_xy / (sum_x2 + 1 / tau2)
    prec_factor = sum_x2 + 1 / tau2

    # Initialize sigma2_beta from the interpretable sigma2_init:
    # conditional variance of beta given sigma^2 is sigma^2 / (sum_x2 + 1/tau2)
    sigma2_beta_init = sigma2_init / prec_factor

    def compute_elbo(sigma2_beta, nu):
        E_log_s2 = jnp.log(nu) - digamma(alpha)
        E_inv_s2 = alpha / nu
        E_beta2 = sigma2_beta + mu_beta**2

        # E_q[log p(y | beta, sigma^2)]
        S = sum_y2 - 2 * mu_beta * sum_xy + E_beta2 * sum_x2
        ll = -n / 2 * jnp.log(2 * jnp.pi) - n / 2 * E_log_s2 - 0.5 * E_inv_s2 * S

        # E_q[log p(beta | sigma^2)]  with prior beta | sigma^2 ~ N(0, tau2 * sigma^2)
        lp_beta = (
            -0.5 * jnp.log(2 * jnp.pi * tau2)
            - 0.5 * E_log_s2
            - 0.5 * E_inv_s2 / tau2 * E_beta2
        )

        # E_q[log p(sigma^2)]  with Jeffreys prior p(sigma^2) ∝ 1/sigma^2
        lp_s2 = -E_log_s2

        # Entropy of q(beta) = N(mu_beta, sigma2_beta)
        q_beta = 0.5 * (1 + jnp.log(2 * jnp.pi * sigma2_beta))

        # Entropy of q(sigma^2) = InvGamma(alpha, nu)
        q_sigma2 = alpha + jnp.log(nu) + gammaln(alpha) - (1 + alpha) * digamma(alpha)

        return ll + lp_beta + lp_s2 + q_beta + q_sigma2

    def step(sigma2_beta, _):
        # E_q(beta)[A(beta)] where A(beta) = sum(y - x*beta)^2 + beta^2/tau2
        E_A = sum_y2 - 2 * mu_beta * sum_xy + (sigma2_beta + mu_beta**2) * prec_factor

        # Update q(sigma^2) = InvGamma(alpha, nu)
        nu = 0.5 * E_A

        # Update q(beta) = N(mu_beta, sigma2_beta)
        # sigma2_beta = 1 / (E[1/sigma^2] * prec_factor) = nu / (alpha * prec_factor)
        sigma2_beta_new = nu / (alpha * prec_factor)

        elbo = compute_elbo(sigma2_beta_new, nu)
        return sigma2_beta_new, (sigma2_beta_new, nu, elbo)

    _, (sigma2_beta_traj, nu_traj, elbo_traj) = jax.lax.scan(
        step, sigma2_beta_init, xs=None, length=max_iter
    )

    return CAVIResult(
        mu_beta=mu_beta,
        sigma2_beta=sigma2_beta_traj,
        nu=nu_traj,
        alpha=alpha,
        elbo=elbo_traj,
    )
