import os
from pathlib import Path
from scripts.tuning_experiment import run_tuning_experiment
from scripts.random_walk import init, build_kernel
from scripts.densities import logdensity_multimodal, logdensity_mvn, logdensity_volcano


OUTPUT_DIR = Path("output/rwmh/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Running tuning experiment for Random Walk Metropolis-Hastings...")
print("- Multivariate Normal distribution...")
run_tuning_experiment(init, build_kernel, logdensity_mvn, OUTPUT_DIR / "tuning_mvn")
print("- Multimodal distribution...")
run_tuning_experiment(
    init, build_kernel, logdensity_multimodal, OUTPUT_DIR / "tuning_multimodal"
)
print("- Volcano distribution...")
run_tuning_experiment(
    init, build_kernel, logdensity_volcano, OUTPUT_DIR / "tuning_volcano"
)
