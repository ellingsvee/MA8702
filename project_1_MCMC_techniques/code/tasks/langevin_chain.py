import os
from pathlib import Path
from scripts.tuning_experiment import run_tuning_experiment
from scripts.langevin import init, build_kernel
from scripts.densities import log_mvn_dist, log_multimodal, log_volcano


OUTPUT_DIR = Path("output/langevin/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running tuning experiment for Multivariate Normal distribution...")
run_tuning_experiment(init, build_kernel, log_mvn_dist, OUTPUT_DIR / "tuning_mvn.svg")
print("Running tuning experiment for Multimodal distribution...")
run_tuning_experiment(init, build_kernel, log_multimodal, OUTPUT_DIR / "tuning_multimodal.svg")

print("Running tuning experiment for Volcano distribution...")
run_tuning_experiment(init, build_kernel, log_volcano, OUTPUT_DIR / "tuning_volcano.svg")