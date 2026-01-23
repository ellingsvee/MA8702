import matplotlib.pyplot as plt
import jax.numpy as jnp

from utils import autocorr


def plot_trace(ax, positions, label):
    ax.plot(positions)
    ax.set_xlabel("Samples")
    ax.set_ylabel(label)


def plot_autocorr(ax, positions, label):
    # positions = states.position

    acf = autocorr(positions)
    ax.bar(range(len(acf)), acf, width=1.0)
    ax.set_ylabel("ACF")
    ax.set_xlabel("Lags")
    # ax.set_title(f"Autocorrelation of {label}")
    ax.set_ylim([-0.2, 1.0])


def plot_traces_and_acf(states, title, output):
    positions = states.position

    fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

    plot_trace(axes[0], positions[:, 0], "x")
    plot_trace(axes[1], positions[:, 1], "y")
    plot_autocorr(axes[2], positions[:, 0], "x")

    axes[1].set_title(title)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_scatter(states, title, output):
    positions = states.position

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    fontsize_title = 20
    fontsize_labels = 18
    fontsize_ticks = 18

    # 2D Scatter plot
    ax.scatter(positions[:, 0], positions[:, 1], alpha=0.5)
    ax.set_xlabel("x", fontsize=fontsize_labels)
    ax.set_ylabel("y", fontsize=fontsize_labels)

    # Set extent to be [-5, 5] in both axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.autoscale(False)

    ax.set_xticks(jnp.arange(-5, 6, 2))
    ax.set_yticks(jnp.arange(-5, 6, 2))
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    ax.set_title(title, fontsize=fontsize_title)

    # ax.axis('equal')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()
