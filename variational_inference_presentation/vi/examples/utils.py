import os

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from scipy.special import loggamma
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
            ax.plot(
                grid, norm.pdf(grid, advi_mu_j, advi_sd_j), linestyle="--", label="ADVI"
            )

        if beta_samples is not None:
            ax.hist(
                np.asarray(beta_samples[:, j]),
                bins=50,
                density=True,
                alpha=1.0,
                label="HMC",
            )

        ax.axvline(
            float(beta_true_np[j]),
            color="red",
            linestyle="--",
            label=rf"$\beta_{{{j + 1}}}^{{\mathrm{{true}}}}$",
        )
        ax.set_xlabel(rf"$\beta_{{{j + 1}}}$")
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


def plot_joint_posterior(
    result: CAVIResult,
    beta_samples: Array,
    sigma2_samples: Array,
    beta_true: float,
    sigma2_true: float,
    x: Array,
    y: Array,
    tau2: float,
    save_path: os.PathLike | None = None,
):
    """Plot 2D joint posterior to demonstrate mean-field assumption limitation.

    Shows:
    - HMC samples (scatter + contours) - captures true correlation
    - CAVI approximation (axis-aligned contours) - mean-field assumption
    - True Normal-inverse-gamma posterior (optional contours)

    This clearly demonstrates that CAVI cannot model correlation between beta and sigma2.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert samples to numpy
    beta_hmc = np.asarray(beta_samples)
    sigma2_hmc = np.asarray(sigma2_samples)

    # Plot HMC samples as scatter
    ax.scatter(beta_hmc, sigma2_hmc, s=1, alpha=0.5, color="gray", label="HMC samples")

    # Create grid for contour plots
    beta_mean = beta_hmc.mean()
    beta_std = beta_hmc.std()
    sigma2_mean = sigma2_hmc.mean()
    sigma2_std = sigma2_hmc.std()

    beta_grid = np.linspace(beta_mean - 4 * beta_std, beta_mean + 4 * beta_std, 200)
    sigma2_grid = np.linspace(
        max(0.01, sigma2_mean - 4 * sigma2_std), sigma2_mean + 4 * sigma2_std, 200
    )
    Beta, Sigma2 = np.meshgrid(beta_grid, sigma2_grid)

    # Compute true Normal-inverse-gamma posterior
    # For conjugate model: p(beta, sigma2 | x, y) is Normal-inverse-gamma
    n = len(x)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    sum_x2 = np.sum(x_arr**2)
    sum_xy = np.sum(x_arr * y_arr)
    sum_y2 = np.sum(y_arr**2)

    # Posterior hyperparameters for Normal-inverse-gamma
    prec_n = sum_x2 + 1.0 / tau2  # posterior precision scaling
    mu_n = sum_xy / prec_n  # posterior mean of beta | sigma2

    # For inverse-gamma part
    alpha_n = (n + 1.0) / 2.0

    # Compute log probability on grid for true posterior
    def true_log_posterior(beta_val, sigma2_val):
        lambda_val = prec_n
        mu_val = sum_xy / prec_n
        gamma_val = 0.5 * (sum_y2 - sum_xy**2 / prec_n)

        logp_true = alpha_n * np.log(gamma_val) - loggamma(alpha_n)
        logp_true += (-alpha_n - 1) * np.log(sigma2_val)
        logp_true -= (2 * gamma_val + lambda_val * (beta_val - mu_val) ** 2) / (
            2 * sigma2_val
        )
        return logp_true

    # Vectorized computation for true posterior
    true_log_prob = np.vectorize(true_log_posterior)(Beta, Sigma2)
    true_prob = np.exp(
        true_log_prob - true_log_prob.max()
    )  # normalize for numerical stability

    # Plot true posterior contours
    levels_true = np.linspace(true_prob.max() * 0.01, true_prob.max() * 0.9, 8)
    contour_true = ax.contour(
        Beta,
        Sigma2,
        true_prob,
        levels=levels_true,
        colors="red",
        linewidths=1.5,
        alpha=1.0,
    )

    # CAVI approximation: q(beta) * q(sigma2) - independent!
    mu_beta = float(result.mu_beta)
    sigma2_beta = float(result.sigma2_beta)
    alpha = float(result.alpha)
    nu = float(result.nu)

    # Compute CAVI densities on grid
    # q(beta) = N(mu_beta, sigma2_beta)
    q_beta = norm.pdf(beta_grid, mu_beta, np.sqrt(sigma2_beta))

    # q(sigma2) = InvGamma(alpha, nu)
    q_sigma2 = invgamma.pdf(sigma2_grid, a=alpha, scale=nu)

    # Joint density under mean-field: q(beta) * q(sigma2)
    # This will produce AXIS-ALIGNED contours (no correlation)
    # Use outer product to get 2D density
    cavi_joint = np.outer(q_sigma2, q_beta)

    # Plot CAVI contours
    levels_cavi = np.linspace(cavi_joint.max() * 0.01, cavi_joint.max() * 0.9, 8)
    contour_cavi = ax.contour(
        Beta,
        Sigma2,
        cavi_joint,
        levels=levels_cavi,
        colors="blue",
        linewidths=1.5,
        linestyles="--",
        alpha=1.0,
    )

    # Add true parameter values
    ax.axvline(
        beta_true,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"True β={beta_true}",
    )
    ax.axhline(
        sigma2_true,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"True σ²={sigma2_true}",
    )
    ax.plot(beta_true, sigma2_true, "k*", markersize=15, label="True parameters")

    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(r"$\sigma^2$", fontsize=12)

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="",
            markersize=5,
            alpha=0.5,
            label="HMC samples",
        ),
        Line2D([0], [0], color="red", linewidth=1.5, label="True posterior"),
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=1.5,
            linestyle="--",
            label="CAVI",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="",
            markersize=10,
            label="True parameters",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
