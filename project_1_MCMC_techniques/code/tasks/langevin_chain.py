import os
from pathlib import Path
from scripts.tuning_experiment import run_tuning_experiment
from scripts.langevin import init, build_kernel
from scripts.densities import logdensity_multimodal, logdensity_mvn, logdensity_volcano


OUTPUT_DIR = Path("output/langevin/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running tuning experiment for Multivariate Normal distribution...")
run_tuning_experiment(init, build_kernel, logdensity_mvn, OUTPUT_DIR / "tuning_mvn.svg")
print("Running tuning experiment for Multimodal distribution...")
run_tuning_experiment(init, build_kernel, logdensity_multimodal, OUTPUT_DIR / "tuning_multimodal.svg")
print("Running tuning experiment for Volcano distribution...")
run_tuning_experiment(init, build_kernel, logdensity_volcano, OUTPUT_DIR / "tuning_volcano.svg")