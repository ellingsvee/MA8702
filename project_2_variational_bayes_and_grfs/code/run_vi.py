from os import PathLike
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
from jax import Array
from jax.scipy.special import digamma, gammaln
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


PATH = Path(__file__).parent
PLT_PATH = PATH / "figures" / "vi"
PLT_PATH.mkdir(parents=True, exist_ok=True)


ALPHA = 0.01
BETA = 0.01
TAU = 1e-6


def plot_elbo_history(
    loglikelihood_history: Array, filename: Union[PathLike, None] = None
):
    plt.plot(loglikelihood_history)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")

    # Adjust figsize and layout
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def generate_dataset(key, N: int = 100) -> Array:
    x = (
        jr.normal(key, (N,)) + 5.0
    )  # Normal distribution with mean 5.0 and standard deviation 1.0
    return x


def compute_elbo(x, alpha_q, beta_q, tau_q, nu_q, priors: tuple):
    alpha, beta, tau = priors

    N = x.shape[0]

    term1 = -0.5 * (1.0 + jnp.log(2 * jnp.pi) - jnp.log(tau_q))
    term2 = (
        jnp.log(beta_q) - gammaln(alpha_q) + (alpha_q - 1) * digamma(alpha_q) - alpha_q
    )

    term3 = 0.5 * N * (
        digamma(alpha_q) - jnp.log(beta_q) - jnp.log(2 * jnp.pi)
    ) - 0.5 * (jnp.sum((x - nu_q) ** 2) + N / tau_q) * (alpha_q / beta_q)
    term4 = 0.5 * (jnp.log(tau) - jnp.log(2 * jnp.pi) - tau * (nu_q**2 + 1 / tau_q))
    term5 = (
        alpha * jnp.log(beta)
        - gammaln(alpha)
        + (alpha - 1) * (digamma(alpha_q) - jnp.log(beta_q))
        - beta * (alpha_q / beta_q)
    )

    return -term1 - term2 + term3 + term4 + term5


def vi(
    x: Array,
    max_iter: int = 1_000,
    tol: float = 1e-8,
    priors: tuple = (ALPHA, BETA, TAU),
):
    alpha, beta, tau = priors

    N = x.shape[0]
    x_bar = jnp.mean(x)

    # # Initialize variational parameters
    alpha_q = alpha + N / 2.0
    beta_q = beta + 0.5 * jnp.sum((x - x_bar) ** 2)
    E_gamma = alpha_q / beta_q  # The expected E[gamma]
    tau_q = E_gamma * N + tau
    nu_q = (E_gamma * N * x_bar) / tau_q

    @jax.jit
    def step(alpha_q, beta_q, tau_q, nu_q):
        # Update alpha_q and beta_q
        alpha_q = alpha + N / 2.0
        beta_q = beta + 0.5 * (jnp.sum((x - nu_q) ** 2) + N / tau_q)

        # Get expected value of gamma
        E_gamma = alpha_q / beta_q

        # Update tau_q and nu_q
        tau_q = E_gamma * N + tau
        nu_q = (E_gamma * N * x_bar) / tau_q

        # Compute the ELBO
        elbo = compute_elbo(x, alpha_q, beta_q, tau_q, nu_q, priors)
        return alpha_q, beta_q, tau_q, nu_q, elbo

    elbo_history = []
    pbar = tqdm(range(max_iter), desc="VI optimization")
    for i in pbar:
        alpha_q, beta_q, tau_q, nu_q, elbo = step(alpha_q, beta_q, tau_q, nu_q)
        elbo_history.append(elbo)
        pbar.set_postfix(ELBO=f"{elbo:.4f}")

    return nu_q, tau_q, alpha_q, beta_q, elbo_history


def mu_posterior_pdf(mu, x, gamma, priors):
    """Compute the true p(mu | x, gamma) which is Gaussian."""
    alpha, beta, tau = priors
    N = x.shape[0]

    mean = gamma * jnp.sum(x) / (gamma * N + tau)
    var = 1.0 / (gamma * N + tau)

    return stats.norm.pdf(mu, loc=mean, scale=jnp.sqrt(var))


def gamma_posterior_pdf(gamma, x, mu, priors):
    """Compute the true p(gamma | x, mu) which is Gamma."""
    alpha, beta, tau = priors
    N = x.shape[0]

    shape = alpha + N / 2.0
    rate = beta + 0.5 * jnp.sum((x - mu) ** 2)

    return stats.gamma.pdf(gamma, a=shape, scale=1.0 / rate)


def log_posterior(mu, gamma, x, priors):
    alpha, beta, tau = priors
    N = x.shape[0]
    return (
        (alpha + N / 2.0 - 1) * jnp.log(gamma)
        - gamma * (beta + 0.5 * jnp.sum(x**2))
        - 0.5 * (gamma * N + tau) * mu**2
        + gamma * jnp.sum(x) * mu
    )


def plot_vi_posteriors(nu_q, tau_q, alpha_q, beta_q, path=None):
    mu_std = 1.0 / jnp.sqrt(tau_q)
    mu_grid = jnp.linspace(nu_q - 4 * mu_std, nu_q + 4 * mu_std, 300)
    q_mu = jnp.exp(stats.norm.logpdf(mu_grid, loc=nu_q, scale=mu_std))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mu_grid, q_mu)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Density")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path / "mu_posterior.svg")
        plt.close(fig)
    else:
        plt.show()

    gamma_mean = alpha_q / beta_q
    gamma_std = jnp.sqrt(alpha_q) / beta_q
    gamma_grid = jnp.linspace(
        max(1e-6, gamma_mean - 4 * gamma_std), gamma_mean + 4 * gamma_std, 300
    )
    q_gamma = jnp.exp(stats.gamma.logpdf(gamma_grid, a=alpha_q, scale=1.0 / beta_q))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gamma_grid, q_gamma)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Density")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path / "gamma_posterior.svg")
        plt.close(fig)
    else:
        plt.show()


def plot_joint_posterior_comparison(
    x, nu_q, tau_q, alpha_q, beta_q, priors, save_path=None
):
    mu_std = 1.0 / jnp.sqrt(tau_q)
    gamma_mean = alpha_q / beta_q
    gamma_std = jnp.sqrt(alpha_q) / beta_q

    mu_grid = jnp.linspace(nu_q - 4 * mu_std, nu_q + 4 * mu_std, 300)
    gamma_grid = jnp.linspace(
        max(1e-6, gamma_mean - 4 * gamma_std), gamma_mean + 4 * gamma_std, 300
    )
    MU, GAMMA = jnp.meshgrid(mu_grid, gamma_grid)

    # True joint posterior (unnormalized, in log-space)
    log_true = jax.vmap(jax.vmap(lambda m, g: log_posterior(m, g, x, priors)))(
        MU, GAMMA
    )
    log_true = log_true - jnp.max(log_true)

    # VI factorized posterior q(mu) * q(gamma)
    log_q_mu = stats.norm.logpdf(MU, loc=nu_q, scale=mu_std)
    log_q_gamma = stats.gamma.logpdf(GAMMA, a=alpha_q, scale=1.0 / beta_q)
    log_q = log_q_mu + log_q_gamma
    log_q = log_q - jnp.max(log_q)

    levels = jnp.linspace(-6, 0, 7)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(
        mu_grid, gamma_grid, log_true, levels=levels, colors="C0", linestyles="-"
    )
    ax.contour(mu_grid, gamma_grid, log_q, levels=levels, colors="C1", linestyles="-")

    ax.plot([], [], color="C0", label="True")
    ax.plot([], [], color="C1", label="VI")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\gamma$")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        # fig.savefig(path / "joint_posterior.svg")
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def run_and_plot_vi(
    dataset: Array, save_path: PathLike, priors: tuple = (ALPHA, BETA, TAU)
):
    nu_q, tau_q, alpha_q, beta_q, elbo_history = vi(dataset, max_iter=10_000)

    plot_elbo_history(elbo_history, filename=PLT_PATH / "elbo_history.svg")
    plot_joint_posterior_comparison(
        dataset,
        nu_q,
        tau_q,
        alpha_q,
        beta_q,
        priors,
        # save_path=PLT_PATH / "joint_posterior.svg",
        save_path=save_path,
    )


if __name__ == "__main__":
    key = jr.key(0)
    dataset = generate_dataset(key)
    dataset_1000_obs = generate_dataset(key, N=1000)

    print("Running VI on dataset with 100 observations...")
    run_and_plot_vi(dataset, save_path=PLT_PATH / "joint_posterior.svg")

    print("Running VI with different priors...")
    run_and_plot_vi(
        dataset,
        save_path=PLT_PATH / "joint_posterior_other_priors.svg",
        priors=(1, 1, 1e-4),
    )

    print("Running VI on dataset with 1000 observations...")
    run_and_plot_vi(
        dataset_1000_obs, save_path=PLT_PATH / "joint_posterior_1000_obs.svg"
    )
