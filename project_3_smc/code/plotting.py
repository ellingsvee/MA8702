from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from jax import Array

SAVE_DIR = Path("figures")
SAVE_DIR.mkdir(exist_ok=True)
SENSORS = np.array([[0.0, 0.0], [40.0, 40.0]])


def _confidence_ellipse(mean, cov, ax, confidence=0.95, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    s = np.sqrt(chi2.ppf(confidence, df=2))
    width, height = 2 * s * np.sqrt(vals)
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
        states[:, 0],
        states[:, 1],
        "b.-",
        markersize=4,
        linewidth=2.5,
        label="Filtered trajectory",
    )

    # Plot confidence ellipses every step
    for i in range(0, len(states), 1):
        _confidence_ellipse(
            states[i, :2],
            covariances[i, :2, :2],
            ax,
            confidence=0.95,
            edgecolor="blue",
            facecolor="none",
            alpha=0.25,
            linewidth=1.5,
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


def plot_joint_filter_map(
    states: list[Array],
    covariances: list[Array],
    labels: list[str],
    save_name: str = "joint_filter_map.svg",
):
    fig, ax = plt.subplots(figsize=(8, 8))

    for state, cov, label in zip(states, covariances, labels):
        state = np.asarray(state)
        cov = np.asarray(cov)

        # Plot trajectory
        ax.plot(
            state[:, 0],
            state[:, 1],
            ".-",
            markersize=4,
            linewidth=2.5,
            label=label,
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
