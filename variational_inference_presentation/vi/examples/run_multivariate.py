from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import vi.mcmc as mcmc
from data import generate_multivariate_data, make_multivariate_logdensity
from tabulate import tabulate
from utils import (
    plot_beta_intervals,
    plot_beta_marginals,
    plot_beta_scatter,
)
from vi.cavi_multivariate import cavi_multivariate

output = Path("output_multivariate")

SEED = 42
TAU2 = 0.25
SIGMA2 = 1.0
N_HMC_SAMPLES = 5_000
GENERATE_PLOTS = True

# Problem sizes to benchmark
SIZES = [
    (500, 10_000),
]


def set_device(device_name):
    if device_name == "GPU":
        devs = jax.devices("gpu")
        if devs:
            jax.config.update("jax_default_device", devs[0])
            return True
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    return device_name == "CPU"


def run_cavi(X, y):
    """Run CAVI with warmup, return (result, elapsed_seconds)."""
    _ = cavi_multivariate(X, y, sigma2_init=SIGMA2, tau2=TAU2)
    jax.block_until_ready(_)

    t0 = time()
    result = cavi_multivariate(X, y, sigma2_init=SIGMA2, tau2=TAU2)
    jax.block_until_ready(result)
    elapsed = time() - t0

    print(f"  CAVI took {elapsed * 1000:.1f} ms")
    return result, elapsed


def run_hmc(X, y, key):
    """Run HMC with warmup, return (states, infos, elapsed_seconds)."""
    P = X.shape[1]
    logdensity_fn = make_multivariate_logdensity(X, y, TAU2)
    init_pos = jnp.zeros(P + 1)
    init_state = mcmc.init(init_pos, logdensity_fn)
    kernel = mcmc.build_kernel(logdensity_fn, step_size=0.001, num_steps=10)

    _ = mcmc.inference_loop(key, kernel, init_state, num_samples=2)
    jax.block_until_ready(_)

    t0 = time()
    states, infos = mcmc.inference_loop(
        key, kernel, init_state, num_samples=N_HMC_SAMPLES
    )
    jax.block_until_ready(states.position)
    elapsed = time() - t0

    print(f"  HMC took {elapsed:.2f} s ({N_HMC_SAMPLES} samples)")
    return states, infos, elapsed


def benchmark(P, N, device_name):
    """Generate data and run both methods at a given problem size."""
    set_device(device_name)

    key = jax.random.key(SEED)
    key, xk, bk, dk, mk = jax.random.split(key, 5)

    X = jax.random.normal(xk, shape=(N, P))
    beta_true = jax.random.normal(bk, shape=(P,)) * 0.5
    y = generate_multivariate_data(dk, X, beta_true, sigma2=SIGMA2)

    cavi_result, cavi_time = run_cavi(X, y)
    states, infos, hmc_time = run_hmc(X, y, mk)

    return {
        "cavi_time": cavi_time,
        "hmc_time": hmc_time,
        "cavi_result": cavi_result,
        "beta_true": beta_true,
        "states": states,
        "infos": infos,
        "accept_rate": float(infos.acceptance_rate.mean()),
    }


def print_tables(results):
    """Print timing comparison and VI-vs-MCMC ratio tables."""
    has_gpu = "GPU" in results

    # Timing table
    rows = []
    for P, N in SIZES:
        row = [f"{P}", f"{N:,}"]
        cpu = results["CPU"][(P, N)]
        row += [f"{cpu['cavi_time'] * 1000:.1f}", f"{cpu['hmc_time']:.2f}"]
        if has_gpu:
            gpu = results["GPU"][(P, N)]
            cavi_speedup = cpu["cavi_time"] / gpu["cavi_time"]
            hmc_speedup = cpu["hmc_time"] / gpu["hmc_time"]
            row += [
                f"{gpu['cavi_time'] * 1000:.1f}",
                f"{gpu['hmc_time']:.2f}",
                f"{cavi_speedup:.1f}x",
                f"{hmc_speedup:.1f}x",
            ]
        rows.append(row)

    headers = ["P", "N", "CAVI CPU (ms)", f"HMC CPU (s, {N_HMC_SAMPLES} samples)"]
    if has_gpu:
        headers += ["CAVI GPU (ms)", "HMC GPU (s)", "CAVI speedup", "HMC speedup"]

    print()
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    # VI vs MCMC ratio table
    ratio_rows = []
    for P, N in SIZES:
        row = [f"{P}", f"{N:,}"]
        cpu = results["CPU"][(P, N)]
        row.append(f"{cpu['hmc_time'] / cpu['cavi_time']:.0f}x")
        if has_gpu:
            gpu = results["GPU"][(P, N)]
            row.append(f"{gpu['hmc_time'] / gpu['cavi_time']:.0f}x")
        ratio_rows.append(row)

    ratio_headers = ["P", "N", "HMC/CAVI (CPU)"]
    if has_gpu:
        ratio_headers.append("HMC/CAVI (GPU)")

    print()
    print(tabulate(ratio_rows, headers=ratio_headers, tablefmt="fancy_grid"))


def generate_plots(results, device):
    """Generate and save comparison plots for the largest problem size."""
    P, N = SIZES[-1]
    r = results[device][(P, N)]

    beta_samples = r["states"].position[:, :P]
    sigma2_samples = jnp.exp(r["states"].position[:, P])

    print("Generating plots...")

    plot_beta_scatter(
        r["cavi_result"],
        beta_true=r["beta_true"],
        beta_samples=beta_samples,
        save_path=output / "beta_scatter.svg",
    )
    plot_beta_intervals(
        r["cavi_result"],
        beta_true=r["beta_true"],
        beta_samples=beta_samples,
        save_path=output / "beta_intervals.svg",
    )
    plot_beta_marginals(
        r["cavi_result"],
        beta_true=r["beta_true"],
        beta_samples=beta_samples,
        save_path=output / "beta_marginals.svg",
    )

    print("Plots saved to:", output)


def main():
    output.mkdir(exist_ok=True)

    has_gpu = bool(jax.devices("gpu"))
    devices = ["CPU", "GPU"] if has_gpu else ["CPU"]

    results = {}
    for dev in devices:
        results[dev] = {}
        for P, N in SIZES:
            print(f"Benchmarking {dev}  P={P}, N={N} ...")
            results[dev][(P, N)] = benchmark(P, N, dev)

    print_tables(results)

    if GENERATE_PLOTS:
        plot_device = "GPU" if has_gpu else "CPU"
        generate_plots(results, plot_device)


if __name__ == "__main__":
    main()
