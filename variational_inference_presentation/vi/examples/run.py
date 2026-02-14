import jax
import jax.numpy as jnp

from time import time

from vi.cavi import cavi
from data import generate_data, make_logdensity
from utils import plot_data, plot_variational_distributions

import vi.mcmc as mcmc

from pathlib import Path

output = Path("output")

SEED = 1
TAU2 = 0.25
BETA = 0.30
SIGMA2 = 1.0
N_SAMPLES = 1_000
GENERATE_PLOTS = True


def main():
    output.mkdir(exist_ok=True)

    key = jax.random.key(SEED)
    key, data_key, mcmc_key = jax.random.split(key, 3)

    x = jnp.linspace(0, 1, N_SAMPLES)
    y = generate_data(data_key, x, beta=BETA, sigma2=SIGMA2)

    # CAVI
    cavi_start_time = time()
    cavi_result = cavi(x, y, sigma2_init=SIGMA2, tau2=TAU2)
    cavi_end_time = time()
    print(f"CAVI took {cavi_end_time - cavi_start_time:.2f} seconds")

    # MCMC (HMC on [beta, log(sigma^2)])
    logdensity_fn = make_logdensity(x, y, TAU2)
    initial_position = jnp.array([BETA, jnp.log(SIGMA2)])
    initial_state = mcmc.init(initial_position, logdensity_fn)

    kernel = mcmc.build_kernel(logdensity_fn, step_size=0.01, num_steps=10)

    mcmc_start_time = time()
    states, infos = mcmc.inference_loop(
        mcmc_key, kernel, initial_state, num_samples=10_000
    )
    mcmc_end_time = time()
    print(f"MCMC took {mcmc_end_time - mcmc_start_time:.2f} seconds")

    if GENERATE_PLOTS:
        beta_samples = states.position[:, 0]
        sigma2_samples = jnp.exp(states.position[:, 1])

        # Only sample 1000 points for plotting
        n_plt_samples = min(1_000, N_SAMPLES)
        plot_data(
            x[:n_plt_samples],
            y[:n_plt_samples],
            beta=BETA,
            save_path=output / "data.svg",
        )
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
