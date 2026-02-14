import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gammaln
from typing import NamedTuple


class CAVIResult(NamedTuple):
    """Result of CAVI for univariate linear regression.

    Variational distributions:
        q(beta) = N(mu_beta, sigma2_beta[-1])
        q(sigma^2) = InvGamma(alpha, nu[-1])
    """

    mu_beta: Array
    sigma2_beta: Array
    nu: Array
    alpha: float
    elbo: Array


@jax.jit(static_argnames=("max_iter", "tol"))
def cavi(
    x,
    y,
    sigma2_init=1.0,
    tau2=0.25,
    max_iter=100,
    tol=1e-6,
):
    n = x.shape[0]

    sum_x2 = jnp.sum(x**2)
    sum_xy = jnp.sum(x * y)
    sum_y2 = jnp.sum(y**2)

    alpha = (n + 1.0) / 2.0
    prec_factor = sum_x2 + 1.0 / tau2
    mu_beta = sum_xy / prec_factor

    def compute_elbo(sigma2_beta, nu):
        E_log_s2 = jnp.log(nu) - digamma(alpha)
        E_inv_s2 = alpha / nu
        E_beta2 = sigma2_beta + mu_beta**2

        S = sum_y2 - 2.0 * mu_beta * sum_xy + E_beta2 * sum_x2

        ll = -n / 2 * jnp.log(2 * jnp.pi)
        ll -= n / 2 * E_log_s2
        ll -= 0.5 * E_inv_s2 * S

        lp_beta = (
            -0.5 * jnp.log(2 * jnp.pi * tau2)
            - 0.5 * E_log_s2
            - 0.5 * E_inv_s2 / tau2 * E_beta2
        )

        lp_s2 = -E_log_s2

        q_beta = 0.5 * (1 + jnp.log(2 * jnp.pi * sigma2_beta))

        q_sigma2 = alpha + jnp.log(nu) + gammaln(alpha) - (1 + alpha) * digamma(alpha)

        return ll + lp_beta + lp_s2 + q_beta + q_sigma2

    def cond_fun(state):
        _, _, elbo, prev_elbo, it = state
        converged = jnp.abs(elbo - prev_elbo) < tol
        return jnp.logical_and(~converged, it < max_iter)

    def body_fun(state):
        sigma2_beta, _, elbo, _, it = state

        # Update parameters
        E_A = sum_y2 - 2 * mu_beta * sum_xy + (sigma2_beta + mu_beta**2) * prec_factor
        nu_new = 0.5 * E_A
        sigma2_beta_new = (alpha / nu_new) / prec_factor

        # For measuring convergence
        elbo_new = compute_elbo(sigma2_beta_new, nu_new)

        return (
            sigma2_beta_new,
            nu_new,
            elbo_new,
            elbo,
            it + 1,
        )

    init_sigma2_beta = sigma2_init / prec_factor
    init_nu = 0.5 * (
        sum_y2 - 2 * mu_beta * sum_xy + (init_sigma2_beta + mu_beta**2) * prec_factor
    )

    init_state = (
        init_sigma2_beta,
        init_nu,
        -jnp.inf,
        -jnp.inf,
        0,
    )

    """
    while cond_fun (ELBO not converged):
        do body_fun
    end
    """
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    _, nu_final, elbo_final, _, _ = final_state

    return CAVIResult(
        mu_beta=mu_beta,
        sigma2_beta=(alpha / nu_final) / prec_factor,
        alpha=alpha,
        nu=nu_final,
        elbo=elbo_final,
    )
