# SHELL SCRIPT FOR RUNNING 
# uv run -m tasks.rwmh_chain tasks/langevin_chain tasks/hmc_chain

uv run -m tasks.rwmh_chain
uv run -m tasks.langevin_chain
uv run -m tasks.hmc_chain