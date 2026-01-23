import os
from pathlib import Path

from scripts.densities import logdensity_multimodal, logdensity_mvn, logdensity_volcano
from scripts.hamiltonian import build_kernel, init
from scripts.tuning_experiment import run_tuning_experiment

OUTPUT_DIR = Path("output/hmc/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running tuning experiment for Hamiltonian Monte Carlo...")

step_sizes = [0.01, 0.05, 0.1, 0.2]

print("- Multivariate Normal distribution...")
run_tuning_experiment(
    init,
    build_kernel,
    logdensity_mvn,
    OUTPUT_DIR / "tuning_mvn",
    sigma_values=step_sizes,
)
print("- Multimodal distribution...")
run_tuning_experiment(
    init,
    build_kernel,
    logdensity_multimodal,
    OUTPUT_DIR / "tuning_multimodal",
    sigma_values=step_sizes,
)
print("- Volcano distribution...")
run_tuning_experiment(
    init,
    build_kernel,
    logdensity_volcano,
    OUTPUT_DIR / "tuning_volcano",
    sigma_values=step_sizes,
)
