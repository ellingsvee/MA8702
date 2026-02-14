import jax.numpy as jnp
from vi.data import generate_data
from utils import plot_data

from pathlib import Path

output = Path("output")

SEED = 1
TAU2 = 0.25
BETA = 0.30
SIGMA2 = 1.0


def main():
    x = jnp.linspace(0, 10, 100)
    y = generate_data(x, beta=BETA, sigma2=SIGMA2, seed=SEED)
    plot_data(x, y, beta=BETA, save_path=output / "data.svg")


if __name__ == "__main__":
    main()
