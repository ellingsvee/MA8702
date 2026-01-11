from os import PathLike
import jax
import jax.numpy as jnp
from typing import Callable

import matplotlib.pyplot as plt

from scripts.utils import autocorr
from scripts.inference import inference_loop

def run_tuning_experiment(init_fn: Callable, build_kernel_fn: Callable, logdensity_fn: Callable, filename: PathLike[str]) -> None:
    """Run RWMH tuning experiment with different proposal stddev (sigma) values
    and plot the results.
    """
    # Run with different sigma values
    sigma_values = [0.1, 0.5, 1.0]
    num_steps = 10000
    burnin = 1000

    key = jax.random.key(42)
    initial_pos = jnp.array([0.0, 0.0])

    fig, axes = plt.subplots(len(sigma_values), 3, figsize=(15, 3 * len(sigma_values)))

    initial_state = init_fn(initial_pos, logdensity_fn)
    for i, sigma in enumerate(sigma_values):
        kernel = build_kernel_fn(logdensity_fn, sigma)

        # Run chain
        key, subkey = jax.random.split(key)
        samples, infos = inference_loop(
            subkey, kernel, initial_state, num_steps
        )


        # Remove burnin
        positions = samples.position[burnin:]
        accept = infos.is_accepted[burnin:]

        # Acceptance rate
        acc_rate = float(jnp.mean(accept))

        row_title = rf"$\sigma = {sigma}$, Accept rate: {acc_rate:.3f}"

        # Trace plots
        axes[i, 0].plot(positions[:, 0], linewidth=0.5)
        axes[i, 0].set_ylabel("Trace x")

        axes[i, 1].plot(positions[:, 1], linewidth=0.5)
        axes[i, 1].set_ylabel("Trace y")
        axes[i, 1].set_title(row_title)

        # Autocorrelation
        acf = autocorr(positions[:, 0])
        axes[i, 2].bar(range(len(acf)), acf, width=1.0)
        axes[i, 2].set_ylabel("ACF")
        axes[i, 2].set_ylim([-0.2, 1.0])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


