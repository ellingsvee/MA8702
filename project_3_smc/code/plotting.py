from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from jax import Array

SAVE_DIR = Path("figures")
SAVE_DIR.mkdir(exist_ok=True)
SENSORS = np.array([[0.0, 0.0], [40.0, 40.0]])


def _confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_filter_map(
    states: Array, covariances: Array, save_name: str = "filter_map.svg"
):
    states = np.asarray(states)
    covariances = np.asarray(covariances)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory
    ax.plot(
        states[:, 0], states[:, 1], "b.-", markersize=4, label="Filtered trajectory"
    )

    # Plot confidence ellipses every 5th step
    for i in range(0, len(states), 1):
        _confidence_ellipse(
            states[i, :2],
            covariances[i, :2, :2],
            ax,
            n_std=2.0,
            edgecolor="blue",
            facecolor="blue",
            alpha=0.1,
        )

    # Plot sensors
    ax.plot(*SENSORS[0], "^", color="black", markersize=12, label="Sensor A")
    ax.plot(*SENSORS[1], "s", color="black", markersize=12, label="Sensor B")

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-5, 45)
    ax.set_ylim(-5, 45)

    fig.tight_layout()

    fig.savefig(SAVE_DIR / save_name)
    plt.close()


def plot_filter_variances(
    states: Array, covariances: Array, save_name: str = "filter_map.svg"
):
    covariances = np.asarray(covariances)
    T = len(covariances)
    time = np.arange(T)

    var_E = covariances[:, 0, 0]
    var_N = covariances[:, 1, 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, var_E, label="East")
    ax.plot(time, var_N, label="North")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Variance (km$^2$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / save_name)
    plt.close()
