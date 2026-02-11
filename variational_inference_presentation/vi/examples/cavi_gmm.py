"""CAVI example: 2D Bayesian Gaussian mixture model on synthetic data."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from vi.cavi import GMMPrior, run_cavi

# ---------------------------------------------------------------------------
# Ground truth and data generation
# ---------------------------------------------------------------------------
TRUE_MU = jnp.array([[-3.0, -1.0], [0.0, 2.0], [3.0, -1.0]])
K = len(TRUE_MU)
N = 300
PRIOR_SIGMA = 5.0

jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)
k1, k2 = jax.random.split(key)

assignments = jax.random.categorical(k1, jnp.zeros(K), shape=(N,))
x = TRUE_MU[assignments] + jax.random.normal(k2, shape=(N, 2))

# ---------------------------------------------------------------------------
# Run CAVI
# ---------------------------------------------------------------------------
prior = GMMPrior(sigma=PRIOR_SIGMA, K=K)
posterior, elbos = run_cavi(x, prior, n_iter=100)

# Sort components by first coordinate of posterior mean
order = jnp.argsort(posterior.m[:, 0])
m_sorted = posterior.m[order]
s2_sorted = posterior.s2[order]

print("=== CAVI results (2D Gaussian mixture) ===")
for k in range(K):
    true_k = TRUE_MU[jnp.argsort(TRUE_MU[:, 0])][k]
    print(
        f"  Component {k + 1}: true mu = ({float(true_k[0]):+.2f}, {float(true_k[1]):+.2f}),  "
        f"posterior m = ({float(m_sorted[k, 0]):+.3f}, {float(m_sorted[k, 1]):+.3f}),  "
        f"posterior std = {float(jnp.sqrt(s2_sorted[k])):.4f}"
    )


# ---------------------------------------------------------------------------
# Helper: draw a confidence ellipse for an isotropic Gaussian
# ---------------------------------------------------------------------------
def draw_ellipse(ax, mean, std, color, label=None, ls="-"):
    """Draw a circle (isotropic Gaussian) at 2-sigma."""
    theta = np.linspace(0, 2 * np.pi, 100)
    cx = float(mean[0]) + 2 * std * np.cos(theta)
    cy = float(mean[1]) + 2 * std * np.sin(theta)
    ax.plot(cx, cy, color=color, lw=2, ls=ls, label=label)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["tab:blue", "tab:orange", "tab:green"]

# 1. ELBO convergence
ax = axes[0]
ax.plot(jnp.arange(1, len(elbos) + 1), elbos, "o-", markersize=3)
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
ax.set_title("ELBO convergence")

# 2. 2D scatter colored by hard assignment + confidence ellipses
ax = axes[1]
hard_assign = jnp.argmax(posterior.phi, axis=1)
for k in range(K):
    mask = hard_assign == k
    ax.scatter(
        x[mask, 0], x[mask, 1],
        c=colors[k], alpha=0.4, s=15, label=f"Cluster {k+1}",
    )

# Inferred component ellipses (2-sigma circles, isotropic)
for k in range(K):
    draw_ellipse(ax, posterior.m[k], float(jnp.sqrt(posterior.s2[k])), colors[k])

# True component means marked with X
for k in range(K):
    ax.scatter(
        TRUE_MU[k, 0], TRUE_MU[k, 1],
        marker="x", s=120, c="black", linewidths=2, zorder=5,
    )

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Data coloured by CAVI assignment")
ax.legend(fontsize=8)
ax.set_aspect("equal")

# 3. Posterior uncertainty contours for each mu_k
ax = axes[2]
for k in range(K):
    mk = posterior.m[k]
    sk = float(jnp.sqrt(posterior.s2[k]))

    # Contour grid centred on posterior mean
    g1 = np.linspace(float(mk[0]) - 4 * sk, float(mk[0]) + 4 * sk, 100)
    g2 = np.linspace(float(mk[1]) - 4 * sk, float(mk[1]) + 4 * sk, 100)
    G1, G2 = np.meshgrid(g1, g2)
    Z = np.exp(-0.5 * ((G1 - float(mk[0]))**2 + (G2 - float(mk[1]))**2) / sk**2)
    ax.contour(G1, G2, Z, levels=[0.05, 0.3, 0.7], colors=[colors[k]], linewidths=1.5)

    # True mean
    ax.scatter(
        TRUE_MU[k, 0], TRUE_MU[k, 1],
        marker="x", s=120, c=colors[k], linewidths=2, zorder=5,
        label=f"True $\\mu_{k+1}$",
    )
    # Posterior mean
    ax.scatter(
        float(mk[0]), float(mk[1]),
        marker="o", s=60, c=colors[k], edgecolors="black", zorder=5,
    )

ax.set_xlabel("$\\mu_1$")
ax.set_ylabel("$\\mu_2$")
ax.set_title("Posterior $q(\\mu_k)$ contours")
ax.legend(fontsize=8)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("output/cavi_gmm.png", dpi=150)
print("Saved output/cavi_gmm.png")
plt.close(fig)
