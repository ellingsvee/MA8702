import os

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from scipy.stats import invgamma, norm
from vi.advi_multivariate import MultivariateADVIResult
from vi.cavi import CAVIResult
from vi.cavi_multivariate import MultivariateCAVIResult


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


def plot_beta_scatter(
    cavi_result: MultivariateCAVIResult,
    beta_true: Array,
    beta_samples: Array | None = None,
    advi_result: MultivariateADVIResult | None = None,
    save_path: os.PathLike | None = None,
):
    """True beta vs posterior mean for CAVI, ADVI, and HMC."""
    fig, ax = plt.subplots(figsize=(5, 5))

    beta_true_np = np.asarray(beta_true)
    mu_np = np.asarray(cavi_result.mu)

    ax.scatter(beta_true_np, mu_np, s=8, alpha=0.6, label="CAVI")
    if advi_result is not None:
        advi_mu = np.asarray(advi_result.mu)
        ax.scatter(beta_true_np, advi_mu, s=8, alpha=0.6, marker="^", label="ADVI")
    if beta_samples is not None:
        hmc_means = np.asarray(beta_samples.mean(axis=0))
        ax.scatter(beta_true_np, hmc_means, s=8, alpha=0.6, marker="x", label="HMC")

    lims = [
        min(beta_true_np.min(), mu_np.min()) - 0.1,
        max(beta_true_np.max(), mu_np.max()) + 0.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel(r"True $\beta_j$")
    ax.set_ylabel(r"Estimated $\beta_j$")
    ax.set_aspect("equal")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_sigma2_posterior(
    cavi_result: MultivariateCAVIResult,
    sigma2_true: float,
    sigma2_samples: Array | None = None,
    save_path: os.PathLike | None = None,
):
    """q(sigma^2) from CAVI vs HMC histogram."""
    fig, ax = plt.subplots(figsize=(5, 4))

    alpha = float(cavi_result.alpha)
    nu = float(cavi_result.nu)
    s2_grid = np.linspace(0.01, nu / (alpha - 1) * 3, 200)
    ax.plot(s2_grid, invgamma.pdf(s2_grid, a=alpha, scale=nu), label=r"$q(\sigma^2)$")

    if sigma2_samples is not None:
        ax.hist(
            np.asarray(sigma2_samples), bins=50, density=True, alpha=0.7, label="HMC"
        )

    ax.axvline(
        sigma2_true, color="red", linestyle="--", label=rf"$\sigma^2 = {sigma2_true}$"
    )
    ax.set_xlabel(r"$\sigma^2$")
    ax.legend()

    # Max and min for xlim
    x_min = min(0.01, sigma2_true * 0.5)
    x_max = max(nu / (alpha - 1) * 3, sigma2_true * 1.5)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_beta_intervals(
    cavi_result: MultivariateCAVIResult,
    beta_true: Array,
    beta_samples: Array | None = None,
    advi_result: MultivariateADVIResult | None = None,
    n_show: int = 20,
    save_path: os.PathLike | None = None,
):
    """95% credible intervals for selected beta components (CAVI, ADVI, HMC)."""
    mu_np = np.asarray(cavi_result.mu)
    sd_np = np.sqrt(np.diag(np.asarray(cavi_result.Sigma)))
    beta_true_np = np.asarray(beta_true)

    p = len(mu_np)
    idx = np.arange(min(n_show, p))
    n_methods = 1 + (advi_result is not None) + (beta_samples is not None)
    offset = 0.2 if n_methods == 3 else 0.15

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(idx) + 1))

    # CAVI 95% intervals
    cavi_lo = mu_np[idx] - 1.96 * sd_np[idx]
    cavi_hi = mu_np[idx] + 1.96 * sd_np[idx]
    ax.errorbar(
        mu_np[idx],
        idx + offset,
        xerr=[mu_np[idx] - cavi_lo, cavi_hi - mu_np[idx]],
        fmt="o",
        markersize=4,
        capsize=3,
        label="CAVI 95% CI",
    )

    if advi_result is not None:
        advi_mu = np.asarray(advi_result.mu)
        advi_sd = np.sqrt(np.diag(np.asarray(advi_result.Sigma)))
        advi_lo = advi_mu[idx] - 1.96 * advi_sd[idx]
        advi_hi = advi_mu[idx] + 1.96 * advi_sd[idx]
        ax.errorbar(
            advi_mu[idx],
            idx,
            xerr=[advi_mu[idx] - advi_lo, advi_hi - advi_mu[idx]],
            fmt="^",
            markersize=4,
            capsize=3,
            label="ADVI 95% CI",
        )

    if beta_samples is not None:
        samples_np = np.asarray(beta_samples[:, idx])
        hmc_lo = np.percentile(samples_np, 2.5, axis=0)
        hmc_hi = np.percentile(samples_np, 97.5, axis=0)
        hmc_mean = samples_np.mean(axis=0)
        ax.errorbar(
            hmc_mean,
            idx - offset,
            xerr=[hmc_mean - hmc_lo, hmc_hi - hmc_mean],
            fmt="s",
            markersize=4,
            capsize=3,
            label="HMC 95% CI",
        )

    ax.scatter(
        beta_true_np[idx],
        idx,
        color="red",
        marker="|",
        s=100,
        zorder=5,
        label=r"True $\beta_j$",
    )

    ax.set_yticks(idx)
    ax.set_yticklabels([rf"$\beta_{{{j + 1}}}$" for j in idx])
    ax.set_xlabel(r"$\beta_j$")
    ax.legend(loc="best")
    ax.invert_yaxis()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_beta_marginals(
    cavi_result: MultivariateCAVIResult,
    beta_true: Array,
    beta_samples: Array | None = None,
    advi_result: MultivariateADVIResult | None = None,
    indices: list[int] | None = None,
    save_path: os.PathLike | None = None,
):
    """Marginal posteriors for selected beta components: CAVI Normal, ADVI Normal, HMC histogram."""
    mu_np = np.asarray(cavi_result.mu)
    sd_np = np.sqrt(np.diag(np.asarray(cavi_result.Sigma)))
    beta_true_np = np.asarray(beta_true)

    if indices is None:
        indices = [0, 1, 2, 3]

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, j in zip(axes, indices):
        mu_j, sd_j = float(mu_np[j]), float(sd_np[j])
        grid = np.linspace(mu_j - 4 * sd_j, mu_j + 4 * sd_j, 200)
        ax.plot(grid, norm.pdf(grid, mu_j, sd_j), label="CAVI")

        if advi_result is not None:
            advi_mu = np.asarray(advi_result.mu)
            advi_sd = np.sqrt(np.diag(np.asarray(advi_result.Sigma)))
            advi_mu_j, advi_sd_j = float(advi_mu[j]), float(advi_sd[j])
            ax.plot(grid, norm.pdf(grid, advi_mu_j, advi_sd_j), linestyle="--", label="ADVI")

        if beta_samples is not None:
            ax.hist(
                np.asarray(beta_samples[:, j]),
                bins=30, density=True, alpha=0.7, label="HMC",
            )

        ax.axvline(
            float(beta_true_np[j]), color="red", linestyle="--",
            label=rf"$\beta_{{{j+1}}}^{{\mathrm{{true}}}}$",
        )
        ax.set_xlabel(rf"$\beta_{{{j+1}}}$")
        ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_elbo_convergence(
    elbo_history: Array,
    save_path: os.PathLike | None = None,
):
    """ADVI ELBO vs iteration."""
    fig, ax = plt.subplots(figsize=(6, 4))
    elbo_np = np.asarray(elbo_history)
    ax.plot(elbo_np, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title("ADVI convergence")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_posterior_sd(
    cavi_result: MultivariateCAVIResult,
    beta_samples: Array,
    save_path: os.PathLike | None = None,
):
    """CAVI posterior SD vs HMC posterior SD for each beta component."""
    cavi_sd = np.sqrt(np.diag(np.asarray(cavi_result.Sigma)))
    hmc_sd = np.asarray(beta_samples.std(axis=0))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(hmc_sd, cavi_sd, s=8, alpha=0.6)

    lims = [0, max(hmc_sd.max(), cavi_sd.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel(r"HMC posterior SD")
    ax.set_ylabel(r"CAVI posterior SD")
    ax.set_aspect("equal")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
