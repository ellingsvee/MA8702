import jax
import jax.numpy as jnp

from time import time

from vi.cavi_multivariate import cavi_multivariate
from data import generate_multivariate_data, make_multivariate_logdensity
from utils import plot_multivariate_comparison

import vi.mcmc as mcmc

from pathlib import Path
from tabulate import tabulate

output = Path("output")

SEED = 42
TAU2 = 0.25
SIGMA2 = 1.0
N_HMC_SAMPLES = 100
GENERATE_PLOTS = True

# Problem sizes to benchmark
SIZES = [
    (500, 10_000),
    # (1000, 20_000),
    # (2000, 40_000),
]


def set_device(device_name):
    if device_name == "GPU":
        devs = jax.devices("gpu")
        if devs:
            jax.config.update("jax_default_device", devs[0])
            return True
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    return device_name == "CPU"


def benchmark(P, N, device_name):
    """Run CAVI and HMC at a given problem size, return timing dict."""
    set_device(device_name)

    key = jax.random.key(SEED)
    key, xk, bk, dk, mk = jax.random.split(key, 5)

    X = jax.random.normal(xk, shape=(N, P))
    beta_true = jax.random.normal(bk, shape=(P,)) * 0.5
    y = generate_multivariate_data(dk, X, beta_true, sigma2=SIGMA2)

    # --- CAVI (warmup then time) ---
    _ = cavi_multivariate(X, y, sigma2_init=SIGMA2, tau2=TAU2)
    jax.block_until_ready(_)

    t0 = time()
    cavi_result = cavi_multivariate(X, y, sigma2_init=SIGMA2, tau2=TAU2)
    jax.block_until_ready(cavi_result)
    cavi_t = time() - t0

    # --- HMC (warmup then time) ---
    logdensity_fn = make_multivariate_logdensity(X, y, TAU2)
    init_pos = jnp.zeros(P + 1)
    init_state = mcmc.init(init_pos, logdensity_fn)
    kernel = mcmc.build_kernel(logdensity_fn, step_size=0.001, num_steps=10)

    _ = mcmc.inference_loop(mk, kernel, init_state, num_samples=2)
    jax.block_until_ready(_)

    t0 = time()
    states, infos = mcmc.inference_loop(
        mk, kernel, init_state, num_samples=N_HMC_SAMPLES
    )
    jax.block_until_ready(states.position)
    hmc_t = time() - t0

    return {
        "cavi_time": cavi_t,
        "hmc_time": hmc_t,
        "cavi_result": cavi_result,
        "beta_true": beta_true,
        "states": states,
        "infos": infos,
        "accept_rate": float(infos.acceptance_rate.mean()),
    }


def main():
    output.mkdir(exist_ok=True)

    has_gpu = bool(jax.devices("gpu"))
    devices = ["CPU", "GPU"] if has_gpu else ["CPU"]

    # results[device][(P,N)] = timing dict
    results = {}
    for dev in devices:
        results[dev] = {}
        for P, N in SIZES:
            print(f"Benchmarking {dev}  P={P}, N={N} ...")
            results[dev][(P, N)] = benchmark(P, N, dev)

    # --- Print comparison table ---
    rows = []
    for P, N in SIZES:
        row = [f"{P}", f"{N:,}"]
        cpu = results["CPU"][(P, N)]
        row += [f"{cpu['cavi_time'] * 1000:.1f}", f"{cpu['hmc_time']:.2f}"]
        if "GPU" in results:
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
    if "GPU" in results:
        headers += ["CAVI GPU (ms)", "HMC GPU (s)", "CAVI speedup", "HMC speedup"]

    print()
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # --- Print VI vs MCMC ratios ---
    print()
    ratio_rows = []
    for P, N in SIZES:
        row = [f"{P}", f"{N:,}"]
        cpu = results["CPU"][(P, N)]
        row.append(f"{cpu['hmc_time'] / cpu['cavi_time']:.0f}x")
        if "GPU" in results:
            gpu = results["GPU"][(P, N)]
            row.append(f"{gpu['hmc_time'] / gpu['cavi_time']:.0f}x")
        ratio_rows.append(row)

    ratio_headers = ["P", "N", "HMC/CAVI (CPU)"]
    if "GPU" in results:
        ratio_headers.append("HMC/CAVI (GPU)")
    print(tabulate(ratio_rows, headers=ratio_headers, tablefmt="github"))

    # --- Optionally plot for the largest size ---
    if GENERATE_PLOTS and "GPU" in results:
        P, N = SIZES[-1]
        r = results["GPU"][(P, N)]
        beta_samples = r["states"].position[:, :P]
        sigma2_samples = jnp.exp(r["states"].position[:, P])
        plot_multivariate_comparison(
            r["cavi_result"],
            beta_true=r["beta_true"],
            sigma2_true=SIGMA2,
            beta_samples=beta_samples,
            sigma2_samples=sigma2_samples,
            save_path=output / "multivariate_comparison.svg",
        )


if __name__ == "__main__":
    main()
