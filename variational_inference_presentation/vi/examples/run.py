import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
import numpy as np

from vi.data import generate_data
from vi.cavi import cavi
from utils import plot_data

from pathlib import Path

output = Path("output")

SEED = 1
TAU2 = 0.25
BETA = 0.30
SIGMA2 = 1.0


def plot_variational_distributions(result, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # q(beta) = N(mu_beta, sigma2_beta)
    mu = float(result.mu_beta)
    sd = float(result.sigma2_beta[-1] ** 0.5)
    beta_grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
    axes[0].plot(beta_grid, norm.pdf(beta_grid, mu, sd))
    axes[0].axvline(BETA, color="red", linestyle="--", label=rf"true $\beta$ = {BETA}")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_title(r"$q(\beta)$")
    axes[0].legend()

    # q(sigma^2) = InvGamma(alpha, nu)
    alpha = float(result.alpha)
    nu = float(result.nu[-1])
    s2_grid = np.linspace(0.01, nu / (alpha - 1) * 3, 200)
    axes[1].plot(s2_grid, invgamma.pdf(s2_grid, a=alpha, scale=nu))
    axes[1].axvline(
        SIGMA2, color="red", linestyle="--", label=rf"true $\sigma^2$ = {SIGMA2}"
    )
    axes[1].set_xlabel(r"$\sigma^2$")
    axes[1].set_title(r"$q(\sigma^2)$")
    axes[1].legend()

    # ELBO convergence
    axes[2].plot(np.arange(1, len(result.elbo) + 1), result.elbo)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("ELBO")
    axes[2].set_title("ELBO convergence")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def main():
    output.mkdir(exist_ok=True)

    x = jnp.linspace(0, 1, 100)
    y = generate_data(x, beta=BETA, sigma2=SIGMA2, seed=SEED)
    plot_data(x, y, beta=BETA, save_path=output / "data.svg")

    result = cavi(x, y, sigma2_init=SIGMA2, tau2=TAU2)
    plot_variational_distributions(
        result, save_path=output / "variational_distributions.svg"
    )


if __name__ == "__main__":
    main()
