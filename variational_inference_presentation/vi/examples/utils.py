import os
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
import numpy as np
from vi.cavi import CAVIResult
from vi.cavi_multivariate import MultivariateCAVIResult

from jax import Array


def plot_data(
    x: Array, y: Array, beta: float | None = None, save_path: os.PathLike | None = None
):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    if beta is not None:
        plt.plot(x, beta * x, color="red")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_variational_distributions(
    result: CAVIResult,
    beta_true: float,
    sigma2_true: float,
    beta_samples: Array | None = None,
    sigma2_samples: Array | None = None,
    save_path=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # q(beta) = N(mu_beta, sigma2_beta)
    mu = float(result.mu_beta)
    sd = float(result.sigma2_beta**0.5)
    beta_grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
    axes[0].plot(beta_grid, norm.pdf(beta_grid, mu, sd), label=r"$q(\beta)$")
    if beta_samples is not None:
        axes[0].hist(
            np.asarray(beta_samples), bins=30, density=True, alpha=1.0, label="HMC"
        )
    axes[0].axvline(
        beta_true, color="red", linestyle="--", label=rf"$\beta$ = {beta_true}"
    )
    axes[0].set_xlabel(r"$\beta$")
    axes[0].legend()

    # q(sigma^2) = InvGamma(alpha, nu)
    alpha = float(result.alpha)
    nu = float(result.nu)
    s2_grid = np.linspace(0.01, nu / (alpha - 1) * 3, 200)
    axes[1].plot(
        s2_grid, invgamma.pdf(s2_grid, a=alpha, scale=nu), label=r"$q(\sigma^2)$"
    )
    if sigma2_samples is not None:
        axes[1].hist(
            np.asarray(sigma2_samples), bins=30, density=True, alpha=1.0, label="HMC"
        )
    axes[1].axvline(
        sigma2_true,
        color="red",
        linestyle="--",
        label=rf"$\sigma^2$ = {sigma2_true}",
    )
    axes[1].set_xlabel(r"$\sigma^2$")
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_multivariate_comparison(
    cavi_result: MultivariateCAVIResult,
    beta_true: Array,
    sigma2_true: float,
    beta_samples: Array | None = None,
    sigma2_samples: Array | None = None,
    save_path: os.PathLike | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    beta_true_np = np.asarray(beta_true)
    mu_np = np.asarray(cavi_result.mu)

    # Panel 1: true beta vs CAVI posterior mean (scatter), with HMC means overlaid
    axes[0].scatter(beta_true_np, mu_np, s=8, alpha=0.6, label="CAVI posterior mean")
    if beta_samples is not None:
        hmc_means = np.asarray(beta_samples.mean(axis=0))
        axes[0].scatter(beta_true_np, hmc_means, s=8, alpha=0.6, marker="x", label="HMC posterior mean")
    lims = [
        min(beta_true_np.min(), mu_np.min()),
        max(beta_true_np.max(), mu_np.max()),
    ]
    axes[0].plot(lims, lims, "r--", linewidth=1, label="y = x")
    axes[0].set_xlabel(r"True $\beta$")
    axes[0].set_ylabel(r"Estimated $\beta$")
    axes[0].legend(fontsize=8)

    # Panel 2: q(sigma^2) vs HMC histogram
    alpha = float(cavi_result.alpha)
    nu = float(cavi_result.nu)
    s2_grid = np.linspace(0.01, nu / (alpha - 1) * 3, 200)
    axes[1].plot(
        s2_grid, invgamma.pdf(s2_grid, a=alpha, scale=nu), label=r"$q(\sigma^2)$"
    )
    if sigma2_samples is not None:
        axes[1].hist(
            np.asarray(sigma2_samples), bins=50, density=True, alpha=0.7, label="HMC"
        )
    axes[1].axvline(
        sigma2_true, color="red", linestyle="--", label=rf"$\sigma^2$ = {sigma2_true}"
    )
    axes[1].set_xlabel(r"$\sigma^2$")
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
