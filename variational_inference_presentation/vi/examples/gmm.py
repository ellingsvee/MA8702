import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from vi.cavi import run_cavi
from vi.bbvi import run_bbvi
from vi.data import generate_gmm_dataset
from vi.gmm import GMMPrior
from utils import tabulate_estimates


K = 4
x, centers = generate_gmm_dataset(K=K, n_samples=300, seed=1)

prior = GMMPrior(sigma=3.0, K=K)

cavi_params = run_cavi(x, prior, n_iter=100)


n, dim = x.shape


def log_joint(mu):
    diff = x[:, None, :] - mu[None, :, :]  # (n, K, dim)
    log_components = (
        -0.5 * dim * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.sum(diff**2, axis=2)
        + jnp.log(
            1.0 / K
        )  # This is because of p(c_i = k) = 1/K for all k, which we marginalise out
    )  # (n, K)
    log_lik = jnp.sum(jax.nn.logsumexp(log_components, axis=1))

    # log p(mu)
    log_prior = jnp.sum(
        -0.5 * dim * jnp.log(2 * jnp.pi * prior.sigma**2)
        - 0.5 * jnp.sum(mu**2, axis=1) / prior.sigma**2
    )

    return log_lik + log_prior


bbvi_params = run_bbvi(
    key=jax.random.key(123),
    log_joint_fn=log_joint,
    shape=(K, dim),
    n_iter=3000,
    n_samples=16,
    lr=0.02,
)


print("CAVI estimates:")
tabulate_estimates(centers, cavi_params)

print("\nBBVI estimates:")
tabulate_estimates(centers, bbvi_params)


# ── Plotting ─────────────────────────────────────────────────────────────────

# COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
BUFFER = 1.0
GRID_SIZE = 150

cavi_m, cavi_s2, cavi_phi = cavi_params
bbvi_mu, bbvi_sigma = bbvi_params

x1_min, x1_max = float(x[:, 0].min()) - BUFFER, float(x[:, 0].max()) + BUFFER
x2_min, x2_max = float(x[:, 1].min()) - BUFFER, float(x[:, 1].max()) + BUFFER


def gmm_log_density(grid_pts, means):
    """Log mixture density: log sum_k 1/K * N(x; means_k, I)."""
    diff = grid_pts[:, None, :] - means[None, :, :]  # (G, K, dim)
    log_comp = -0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(diff**2, axis=2)
    return jax.nn.logsumexp(log_comp + jnp.log(1.0 / K), axis=1)


# ── Plot 1: Raw data with true centres ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x[:, 0], x[:, 1], s=15, c="gray", alpha=0.5, label="Data")
for k in range(K):
    ax.plot(
        centers[k, 0],
        centers[k, 1],
        "*",
        markersize=18,
        # color=COLORS[k],
        markeredgecolor="black",
        markeredgewidth=1,
        label=f"True $\\mu_{k}$",
    )
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("gmm_data.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Plot 2: CAVI cluster assignments + posterior centres ─────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
assignments = jnp.argmax(cavi_phi, axis=1)
for k in range(K):
    mask = assignments == k
    ax.scatter(x[mask, 0], x[mask, 1], s=15, alpha=0.4)
    ax.plot(
        cavi_m[k, 0],
        cavi_m[k, 1],
        "*",
        markersize=18,
        # color=COLORS[k],
        markeredgecolor="black",
        markeredgewidth=1,
    )
    r = 2 * float(jnp.sqrt(cavi_s2[k]))
    ax.add_patch(
        plt.Circle(
            (float(cavi_m[k, 0]), float(cavi_m[k, 1])),
            r,
            fill=False,
            # color=COLORS[k],
            linewidth=2,
            linestyle="--",
        )
    )
ax.set_title("CAVI Posterior", fontsize=12, fontweight="bold")
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("gmm_cavi.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Plot 3: BBVI cluster assignments + posterior centres ─────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
dists = jnp.sum((x[:, None, :] - bbvi_mu[None, :, :]) ** 2, axis=2)
bbvi_assignments = jnp.argmin(dists, axis=1)
for k in range(K):
    mask = bbvi_assignments == k
    ax.scatter(x[mask, 0], x[mask, 1], s=15, alpha=0.4)
    ax.plot(
        bbvi_mu[k, 0],
        bbvi_mu[k, 1],
        "*",
        markersize=18,
        # color=COLORS[k],
        markeredgecolor="black",
        markeredgewidth=1,
    )
    ax.add_patch(
        Ellipse(
            (float(bbvi_mu[k, 0]), float(bbvi_mu[k, 1])),
            width=4 * float(bbvi_sigma[k, 0]),
            height=4 * float(bbvi_sigma[k, 1]),
            fill=False,
            # color=COLORS[k],
            linewidth=2,
            linestyle="--",
        )
    )
ax.set_title("BBVI Posterior", fontsize=12, fontweight="bold")
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("gmm_bbvi.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Plot 4: CAVI predictive density ──────────────────────────────────────────

x1_grid = jnp.linspace(x1_min, x1_max, GRID_SIZE)
x2_grid = jnp.linspace(x2_min, x2_max, GRID_SIZE)
X1, X2 = jnp.meshgrid(x1_grid, x2_grid, indexing="ij")
grid_pts = jnp.stack([X1.ravel(), X2.ravel()], axis=1)

cavi_density = jnp.exp(gmm_log_density(grid_pts, cavi_m)).reshape(GRID_SIZE, GRID_SIZE)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.contourf(X1, X2, cavi_density, levels=20, cmap="viridis", alpha=0.85)
ax.scatter(
    x[:, 0], x[:, 1], s=10, c="white", edgecolors="gray", alpha=0.5, linewidths=0.5
)
for k in range(K):
    ax.plot(
        cavi_m[k, 0],
        cavi_m[k, 1],
        "*",
        markersize=18,
        color="red",
        markeredgecolor="black",
        markeredgewidth=1,
    )
plt.colorbar(c, ax=ax, label="$p(x)$")
ax.set_title("CAVI Predictive Density", fontsize=12, fontweight="bold")
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("gmm_cavi_density.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Plot 5: BBVI predictive density ──────────────────────────────────────────

bbvi_density = jnp.exp(gmm_log_density(grid_pts, bbvi_mu)).reshape(GRID_SIZE, GRID_SIZE)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.contourf(X1, X2, bbvi_density, levels=20, cmap="viridis", alpha=0.85)
ax.scatter(
    x[:, 0], x[:, 1], s=10, c="white", edgecolors="gray", alpha=0.5, linewidths=0.5
)
for k in range(K):
    ax.plot(
        bbvi_mu[k, 0],
        bbvi_mu[k, 1],
        "*",
        markersize=18,
        color="red",
        markeredgecolor="black",
        markeredgewidth=1,
    )
plt.colorbar(c, ax=ax, label="$p(x)$")
ax.set_title("BBVI Predictive Density", fontsize=12, fontweight="bold")
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("gmm_bbvi_density.png", dpi=150, bbox_inches="tight")
plt.close()
