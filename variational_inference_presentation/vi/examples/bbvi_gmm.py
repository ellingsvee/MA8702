"""BBVI example: 2D Bayesian Gaussian mixture model on synthetic data.

The discrete cluster assignments c_i are marginalised out so that the
log-joint is smooth in mu and BBVI with the reparameterisation trick
applies directly.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from vi.bbvi import run_bbvi

# ---------------------------------------------------------------------------
# Ground truth and data generation (same seed/params as CAVI example)
# ---------------------------------------------------------------------------
TRUE_MU = jnp.array([[-3.0, -1.0], [0.0, 2.0], [3.0, -1.0]])
K, D = TRUE_MU.shape
N = 300
PRIOR_SIGMA = 5.0

jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)
k1, k2 = jax.random.split(key)

assignments = jax.random.categorical(k1, jnp.zeros(K), shape=(N,))
x = TRUE_MU[assignments] + jax.random.normal(k2, shape=(N, D))

print(f"Data: {N} points, K={K} clusters, D={D}")

# ---------------------------------------------------------------------------
# Log-joint with c_i marginalised out
# log p(x, mu) = sum_i log[ sum_k (1/K) N(x_i | mu_k, I_D) ] + sum_k log N(mu_k | 0, sigma^2 I_D)
# ---------------------------------------------------------------------------


def log_joint(mu_flat):
    mu = mu_flat.reshape(K, D)  # (K, D)

    # log p(x | mu) with c marginalised
    # = sum_i log sum_k exp( log(1/K) + log N(x_i | mu_k, I_D) )
    diff = x[:, None, :] - mu[None, :, :]  # (N, K, D)
    log_components = (
        -0.5 * D * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.sum(diff**2, axis=2)
        + jnp.log(1.0 / K)
    )  # (N, K)
    log_lik = jnp.sum(jax.nn.logsumexp(log_components, axis=1))

    # log p(mu) = sum_k log N(mu_k | 0, sigma^2 I_D)
    log_prior = jnp.sum(
        -0.5 * D * jnp.log(2 * jnp.pi * PRIOR_SIGMA**2)
        - 0.5 * jnp.sum(mu**2, axis=1) / PRIOR_SIGMA**2
    )

    return log_lik + log_prior


# ---------------------------------------------------------------------------
# Run BBVI
# ---------------------------------------------------------------------------
key_bbvi = jax.random.key(123)
result = run_bbvi(
    key_bbvi,
    log_joint,
    dim=K * D,  # mu is flattened to (K*D,) = (6,)
    n_iter=3000,
    n_samples=16,
    lr=0.02,
)

# Reshape results back to (K, D)
mu_inferred = result.mu.reshape(K, D)
sigma_inferred = result.sigma.reshape(K, D)

# Sort components by first coordinate
order = jnp.argsort(mu_inferred[:, 0])
mu_sorted = mu_inferred[order]
sigma_sorted = sigma_inferred[order]
true_sorted = TRUE_MU[jnp.argsort(TRUE_MU[:, 0])]

print("\n=== BBVI results (2D Gaussian mixture) ===")
for k in range(K):
    print(
        f"  Component {k+1}: true mu = ({float(true_sorted[k, 0]):+.2f}, {float(true_sorted[k, 1]):+.2f}),  "
        f"posterior mean = ({float(mu_sorted[k, 0]):+.3f}, {float(mu_sorted[k, 1]):+.3f}),  "
        f"posterior std = ({float(sigma_sorted[k, 0]):.4f}, {float(sigma_sorted[k, 1]):.4f})"
    )


# ---------------------------------------------------------------------------
# Helper: assign data points to nearest inferred component
# ---------------------------------------------------------------------------
def assign_points(x, mu):
    """Hard assignment based on Euclidean distance to inferred means."""
    dists = jnp.sum((x[:, None, :] - mu[None, :, :]) ** 2, axis=2)
    return jnp.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["tab:blue", "tab:orange", "tab:green"]

# 1. ELBO convergence (raw + smoothed)
ax = axes[0]
elbos = result.elbos
iters = jnp.arange(1, len(elbos) + 1)
ax.plot(iters, elbos, alpha=0.25, color="steelblue", label="Raw ELBO")
window = 50
smoothed = jnp.convolve(elbos, jnp.ones(window) / window, mode="valid")
ax.plot(jnp.arange(window, len(elbos) + 1), smoothed, color="darkblue", lw=2, label=f"Smoothed ({window})")
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
ax.set_title("ELBO convergence")
ax.legend(fontsize=8)

# 2. 2D scatter colored by hard assignment + confidence ellipses
ax = axes[1]
hard_assign = assign_points(x, mu_inferred)
for k in range(K):
    mask = hard_assign == k
    ax.scatter(
        x[mask, 0], x[mask, 1],
        c=colors[k], alpha=0.4, s=15, label=f"Cluster {k+1}",
    )

# Inferred component ellipses (2-sigma, using per-dimension sigma)
for k in range(K):
    theta = np.linspace(0, 2 * np.pi, 100)
    cx = float(mu_inferred[k, 0]) + 2 * float(sigma_inferred[k, 0]) * np.cos(theta)
    cy = float(mu_inferred[k, 1]) + 2 * float(sigma_inferred[k, 1]) * np.sin(theta)
    ax.plot(cx, cy, color=colors[k], lw=2)

# True means marked with X
for k in range(K):
    ax.scatter(
        TRUE_MU[k, 0], TRUE_MU[k, 1],
        marker="x", s=120, c="black", linewidths=2, zorder=5,
    )

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Data coloured by BBVI assignment")
ax.legend(fontsize=8)
ax.set_aspect("equal")

# 3. Posterior uncertainty contours for each mu_k
ax = axes[2]
for k in range(K):
    mk = mu_inferred[k]
    sk = sigma_inferred[k]  # (D,) per-dimension sigma

    # Contour grid
    sx, sy = float(sk[0]), float(sk[1])
    g1 = np.linspace(float(mk[0]) - 4 * sx, float(mk[0]) + 4 * sx, 100)
    g2 = np.linspace(float(mk[1]) - 4 * sy, float(mk[1]) + 4 * sy, 100)
    G1, G2 = np.meshgrid(g1, g2)
    Z = np.exp(
        -0.5 * ((G1 - float(mk[0]))**2 / sx**2 + (G2 - float(mk[1]))**2 / sy**2)
    )
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
fig.savefig("output/bbvi_gmm.png", dpi=150)
print("Saved output/bbvi_gmm.png")
plt.close(fig)
