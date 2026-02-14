import jax
import jax.numpy as jnp

from vi.data import generate_data
from vi.cavi import cavi
from utils import plot_data, plot_variational_distributions

import vi.mcmc as mcmc

from pathlib import Path

output = Path("output")

SEED = 1
TAU2 = 0.25
BETA = 0.30
SIGMA2 = 1.0


def make_logdensity(x, y, tau2):
    n = x.shape[0]

    def logdensity_fn(params):
        beta, log_sigma2 = params[0], params[1]
        sigma2 = jnp.exp(log_sigma2)

        # log p(y | beta, sigma^2)
        residuals = y - x * beta
        ll = -n / 2 * jnp.log(2 * jnp.pi * sigma2) - jnp.sum(residuals**2) / (
            2 * sigma2
        )

        # log p(beta | sigma^2) ~ N(0, tau2 * sigma^2)
        lp_beta = -0.5 * jnp.log(2 * jnp.pi * tau2 * sigma2) - beta**2 / (
            2 * tau2 * sigma2
        )

        # Jeffreys prior p(sigma^2) ∝ 1/sigma^2 cancels with Jacobian of log-transform
        return ll + lp_beta

    return logdensity_fn


def main():
    output.mkdir(exist_ok=True)

    key = jax.random.key(SEED)
    key, data_key, mcmc_key = jax.random.split(key, 3)

    x = jnp.linspace(0, 1, 100)
    y = generate_data(data_key, x, beta=BETA, sigma2=SIGMA2)
    plot_data(x, y, beta=BETA, save_path=output / "data.svg")

    # CAVI
    cavi_result = cavi(x, y, sigma2_init=SIGMA2, tau2=TAU2)

    # MCMC (HMC on [beta, log(sigma^2)])
    logdensity_fn = make_logdensity(x, y, TAU2)
    initial_position = jnp.array([BETA, jnp.log(SIGMA2)])
    initial_state = mcmc.init(initial_position, logdensity_fn)

    kernel = mcmc.build_kernel(logdensity_fn, step_size=0.01, num_steps=10)
    states, infos = mcmc.inference_loop(
        mcmc_key, kernel, initial_state, num_samples=10_000
    )

    beta_samples = states.position[:, 0]
    sigma2_samples = jnp.exp(states.position[:, 1])

    plot_variational_distributions(
        cavi_result,
        beta_true=BETA,
        sigma2_true=SIGMA2,
        beta_samples=beta_samples,
        sigma2_samples=sigma2_samples,
        save_path=output / "variational_distributions.svg",
    )


if __name__ == "__main__":
    main()
