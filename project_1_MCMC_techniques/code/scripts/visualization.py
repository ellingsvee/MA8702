import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from densities import log_mvn_dist, multimodal, log_volcano

def plot_distribution_heatmat(ax, dist: Array, title: str) -> None: 
    """Plot a heatmap of the given distribution on the provided axis.

    Args:
        ax: The matplotlib axis to plot on.
        dist: A 2D array representing the distribution values.
        title: The title for the plot.
    """
    heatmap = ax.imshow(dist, origin='lower', cmap='viridis', extent=(-5, 5, -5, 5))

    fontsize_title = 20
    fontsize_labels = 18
    fontsize_ticks = 18

    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel('X-axis', fontsize=fontsize_labels)
    ax.set_ylabel('Y-axis', fontsize=fontsize_labels)

    ax.set_xticks(jnp.arange(-5, 6, 2))
    ax.set_yticks(jnp.arange(-5, 6, 2))
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.02)
    # cbar.set_label('Probability Density', fontsize=14)
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    
    plt.tight_layout()

if __name__ == "__main__":
    # Create grid [-5, 5] x [-5, 5] with 0.1 spacing
    x = jnp.arange(-5, 5.1, 0.1)
    X, Y = jnp.meshgrid(x, x)
    grid_points = jnp.column_stack([X.ravel(), Y.ravel()])

    # Compute the distributions values
    Z_MVN = jnp.exp(log_mvn_dist(grid_points).reshape(X.shape))
    Z_multimodal = multimodal(grid_points).reshape(X.shape)
    Z_volcano = jnp.exp(log_volcano(grid_points).reshape(X.shape))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(22, 6))
    plot_distribution_heatmat(axs[0], Z_MVN, "Multivariate Normal")
    plot_distribution_heatmat(axs[1], Z_multimodal, "Multimodal Distribution")
    plot_distribution_heatmat(axs[2], Z_volcano, "Volcano Distribution")
    
    # Save the figure
    plt.savefig("output/distributions.svg")
    plt.close()