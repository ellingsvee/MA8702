import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt

from scripts.utils import autocorr
from scripts.rwmh import run_chain

def run_tuning_experiment(logdensity_fn: Callable, filename: str) -> None:
    """Run RWMH tuning experiment with different proposal stddev (sigma) values
    and plot the results.
    """
    # Run with different sigma values
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    num_steps = 10000
    warmup = 1000

    key = jax.random.key(42)
    initial_pos = jnp.array([0.0, 0.0])

    fig, axes = plt.subplots(len(sigma_values), 3, figsize=(15, 3*len(sigma_values)))

    for i, sigma in enumerate(sigma_values):
        print(f"\nsigma = {sigma}")

        # Run chain
        key, subkey = jax.random.split(key)
        samples, _, accept = run_chain(subkey, initial_pos, logdensity_fn, sigma, num_steps)

        # Remove warmup
        samples = samples[warmup:]
        accept = accept[warmup:]

        # Acceptance rate
        acc_rate = float(jnp.mean(accept))
        print(f"  Acceptance rate: {acc_rate:.3f}")

        row_title = rf"$\sigma = {sigma}$, Accept rate: {acc_rate:.3f}"

                # Trace plots
        axes[i, 0].plot(samples[:, 0], linewidth=0.5)
        axes[i, 0].set_ylabel(f'Trace x')

        axes[i, 1].plot(samples[:, 1], linewidth=0.5)
        axes[i, 1].set_ylabel(f'Trace y')
        axes[i, 1].set_title(row_title)

        # Autocorrelation
        acf = autocorr(samples[:, 0])
        axes[i, 2].bar(range(len(acf)), acf, width=1.0)
        axes[i, 2].set_ylabel('ACF')
        axes[i, 2].set_ylim([-0.2, 1.0])


    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    from scripts.densities import log_mvn_dist, log_multimodal, log_volcano

    print("Running tuning experiment for Multivariate Normal distribution...")
    run_tuning_experiment(log_mvn_dist, "output/rwmh_tuning_mvn.svg")

    print("Running tuning experiment for Multimodal distribution...")
    run_tuning_experiment(log_multimodal, "output/rwmh_tuning_multimodal.svg")

    print("Running tuning experiment for Volcano distribution...")
    run_tuning_experiment(log_volcano, "output/rwmh_tuning_volcano.svg")