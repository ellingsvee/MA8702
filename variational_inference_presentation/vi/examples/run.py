from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
from mcmc import hamiltonian, inference_loop
from data import generate_data, make_logdensity
from jax import Array
from utils import plot_data, plot_variational_distributions
from vi.cavi import CAVIResult, cavi

output = Path("output")

SEED = 1
TAU2 = 0.25
BETA = 0.30
SIGMA2 = 1.0
# N_SAMPLES = 1_000_000
N_SAMPLES = 1_000
GENERATE_PLOTS = True
DEVICE = "CPU"  # or "GPU"


def set_device(device_name):
    if device_name == "GPU":
        devs = jax.devices("gpu")
        if devs:
            jax.config.update("jax_default_device", devs[0])
            return True
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    return device_name == "CPU"


def run_cavi(x: Array, y: Array) -> CAVIResult:
    # CAVI
    cavi_start_time = time()
    cavi_result = cavi(x, y, sigma2_init=SIGMA2, tau2=TAU2)
    cavi_end_time = time()
    print(f"CAVI took {cavi_end_time - cavi_start_time:.2f} seconds")
    return cavi_result


def run_mcmc(x: Array, y: Array, mcmc_key: Array):
    logdensity_fn = make_logdensity(x, y, TAU2)
    initial_position = jnp.array([BETA, jnp.log(SIGMA2)])
    initial_state = hamiltonian.init(initial_position, logdensity_fn)
    kernel = hamiltonian.build_kernel(logdensity_fn, step_size=0.01, num_steps=10)

    _ = inference_loop(mcmc_key, kernel, initial_state, num_samples=2)
    jax.block_until_ready(_)

    # Run chain
    mcmc_start_time = time()
    states, _ = inference_loop(mcmc_key, kernel, initial_state, num_samples=10_000)
    jax.block_until_ready(states.position)
    mcmc_end_time = time()
    print(f"MCMC took {mcmc_end_time - mcmc_start_time:.2f} seconds")
    return states


def main():
    set_device(DEVICE)
    output.mkdir(exist_ok=True)

    key = jax.random.key(SEED)
    key, data_key, mcmc_key = jax.random.split(key, 3)

    x = jnp.linspace(0, 10, N_SAMPLES)
    y = generate_data(data_key, x, beta=BETA, sigma2=SIGMA2)

    cavi_result = run_cavi(x, y)
    states = run_mcmc(x, y, mcmc_key)

    if GENERATE_PLOTS:
        print("Generating plots...")
        beta_samples = states.position[:, 0]
        sigma2_samples = jnp.exp(states.position[:, 1])

        # Only sample 1000 points for plotting
        n_plt_samples = min(1_000, N_SAMPLES)
        plot_data(
            x,
            y,
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
        print("Plots saved to:", output)


if __name__ == "__main__":
    main()
