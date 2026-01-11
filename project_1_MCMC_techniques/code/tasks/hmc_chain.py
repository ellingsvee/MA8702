import os
from pathlib import Path
from scripts.tuning_experiment import run_tuning_experiment
from scripts.hamiltonian import init, build_kernel
from scripts.densities import logdensity_multimodal, logdensity_mvn, logdensity_volcano


OUTPUT_DIR = Path("output/hmc/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running tuning experiment for Hamiltonian Monte Carlo...")
print("- Multivariate Normal distribution...")
run_tuning_experiment(init, build_kernel, logdensity_mvn, OUTPUT_DIR / "tuning_mvn")
print("- Multimodal distribution...")
run_tuning_experiment(init, build_kernel, logdensity_multimodal, OUTPUT_DIR / "tuning_multimodal")
print("- Volcano distribution...")
run_tuning_experiment(init, build_kernel, logdensity_volcano, OUTPUT_DIR / "tuning_volcano")