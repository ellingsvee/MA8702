import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gammaln
from typing import NamedTuple


class MultivariateCAVIResult(NamedTuple):
    """Result of CAVI for multivariate linear regression.

    Model:
        y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2 * I_n)
        beta | sigma^2 ~ N(0, tau^2 * sigma^2 * I_p)
        p(sigma^2) proportional to 1/sigma^2  (Jeffrey's prior)

    Variational distributions:
        q(beta) = N(mu, Sigma)
        q(sigma^2) = InvGamma(alpha, nu)
    """

    mu: Array
    Sigma: Array
    alpha: float
    nu: Array
    elbo: Array


@jax.jit(static_argnames=("max_iter", "tol"))
def cavi_multivariate(
    X,
    y,
    sigma2_init=1.0,
    tau2=0.25,
    max_iter=100,
    tol=1e-6,
):
    n, p = X.shape

    # Precompute sufficient statistics (GPU-friendly dense linear algebra)
    XtX = X.T @ X  # (p, p)
    Xty = X.T @ y  # (p,)

    alpha = (n + p) / 2.0

    # Precision factor: constant across iterations
    P = XtX + (1.0 / tau2) * jnp.eye(p)  # (p, p)

    # Posterior mean: constant across iterations
    mu = jnp.linalg.solve(P, Xty)  # (p,)

    # Log-determinant of P (for ELBO)
    _, logdet_P = jnp.linalg.slogdet(P)

    # Precompute P^{-1} for covariance and trace computation
    P_inv = jnp.linalg.solve(P, jnp.eye(p))  # (p, p)
    tr_P_inv = jnp.trace(P_inv)

    # Precompute trace(XtX @ P_inv) for nu update
    tr_XtX_Pinv = jnp.trace(XtX @ P_inv)

    # Residual sum of squares at the mean
    residual = y - X @ mu
    rss = residual @ residual  # = yty - 2*mu@Xty + mu@XtX@mu

    def compute_elbo(nu):
        E_log_s2 = jnp.log(nu) - digamma(alpha)
        E_inv_s2 = alpha / nu

        # E[||y - X beta||^2] = rss + trace(XtX @ Sigma)
        # where Sigma = (nu/alpha) * P_inv
        E_residual = rss + (nu / alpha) * tr_XtX_Pinv

        # E[beta^T beta] = mu^T mu + trace(Sigma)
        tr_Sigma = (nu / alpha) * jnp.trace(P_inv)
        E_beta2 = mu @ mu + tr_Sigma

        # Log-likelihood: E_q[-n/2 log(2pi) - n/2 log(sigma^2) - 1/(2 sigma^2) ||y - X beta||^2]
        ll = -n / 2 * jnp.log(2 * jnp.pi)
        ll -= n / 2 * E_log_s2
        ll -= 0.5 * E_inv_s2 * E_residual

        # Log-prior on beta: E_q[-p/2 log(2pi tau^2) - p/2 log(sigma^2) - 1/(2 tau^2 sigma^2) beta^T beta]
        lp_beta = -p / 2 * jnp.log(2 * jnp.pi * tau2)
        lp_beta -= p / 2 * E_log_s2
        lp_beta -= 0.5 * E_inv_s2 / tau2 * E_beta2

        # Log-prior on sigma^2 (Jeffrey's): -log(sigma^2)
        lp_s2 = -E_log_s2

        # Entropy of q(beta) = N(mu, Sigma): p/2 (1 + log(2pi)) + 1/2 log|Sigma|
        # log|Sigma| = log|(nu/alpha) P^{-1}| = p*log(nu/alpha) - logdet_P
        log_det_Sigma = p * jnp.log(nu / alpha) - logdet_P
        q_beta = 0.5 * (p * (1 + jnp.log(2 * jnp.pi)) + log_det_Sigma)

        # Entropy of q(sigma^2) = InvGamma(alpha, nu)
        q_sigma2 = alpha + jnp.log(nu) + gammaln(alpha) - (1 + alpha) * digamma(alpha)

        return ll + lp_beta + lp_s2 + q_beta + q_sigma2

    def cond_fun(state):
        nu, elbo, prev_elbo, it = state
        converged = jnp.abs(elbo - prev_elbo) < tol
        return jnp.logical_and(~converged, it < max_iter)

    def body_fun(state):
        nu, elbo, prev_elbo, it = state

        # Simpler derivation: nu = 0.5 * E[||y - X beta||^2 + beta^T beta / tau^2]
        # Using current q(sigma^2) to get Sigma = (nu/alpha)*P_inv:
        E_residual = rss + (nu / alpha) * tr_XtX_Pinv
        E_beta2 = mu @ mu + (nu / alpha) * tr_P_inv

        nu_new = 0.5 * (E_residual + E_beta2 / tau2)

        elbo_new = compute_elbo(nu_new)

        return (nu_new, elbo_new, elbo, it + 1)

    # Initialize
    init_E_inv_s2 = 1.0 / sigma2_init
    init_nu = alpha / init_E_inv_s2  # so that alpha/nu = 1/sigma2_init

    init_state = (init_nu, -jnp.inf, -jnp.inf, 0)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    nu_final, elbo_final, _, _ = final_state

    Sigma = (nu_final / alpha) * P_inv

    return MultivariateCAVIResult(
        mu=mu,
        Sigma=Sigma,
        alpha=alpha,
        nu=nu_final,
        elbo=elbo_final,
    )
