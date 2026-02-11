"""BBVI example: Bayesian logistic regression on synthetic 2-D data."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm

from vi.bbvi import run_bbvi

# ---------------------------------------------------------------------------
# Synthetic data: 2-D features, binary labels
# ---------------------------------------------------------------------------
TRUE_W = jnp.array([2.0, -1.5])  # true weights (no bias for simplicity)
N = 200

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)

X = jax.random.normal(k1, shape=(N, 2))
logits = X @ TRUE_W
probs = jax.nn.sigmoid(logits)
y = jax.random.bernoulli(k2, probs).astype(jnp.float64)

print(f"Data: {N} points, {int(y.sum())} positive")

# ---------------------------------------------------------------------------
# Log-joint: log p(y, w) = log p(y | X, w) + log p(w)
# ---------------------------------------------------------------------------
PRIOR_SCALE = 5.0  # N(0, 5^2) prior on each weight


def log_joint(w):
    logits = X @ w
    log_lik = jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
    log_prior = -0.5 * jnp.sum((w / PRIOR_SCALE) ** 2) - w.shape[0] * jnp.log(PRIOR_SCALE)
    return log_lik + log_prior


# ---------------------------------------------------------------------------
# Run BBVI
# ---------------------------------------------------------------------------
result = run_bbvi(
    key,
    log_joint,
    dim=2,
    n_iter=3000,
    n_samples=16,
    lr=0.02,
)

print("\n=== BBVI results ===")
print(f"  True weights:      {TRUE_W}")
print(f"  Posterior mean:     {result.mu}")
print(f"  Posterior std:      {result.sigma}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. ELBO convergence (raw + smoothed)
ax = axes[0]
elbos = result.elbos
iters = jnp.arange(1, len(elbos) + 1)
ax.plot(iters, elbos, alpha=0.25, color="steelblue", label="Raw ELBO")
# running average
window = 50
smoothed = jnp.convolve(elbos, jnp.ones(window) / window, mode="valid")
ax.plot(jnp.arange(window, len(elbos) + 1), smoothed, color="darkblue", lw=2, label=f"Smoothed ({window})")
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
ax.set_title("ELBO convergence")
ax.legend(fontsize=8)

# 2. Posterior predictive decision boundary
ax = axes[1]
colors = ["tab:blue" if yi == 0 else "tab:red" for yi in y]
ax.scatter(X[:, 0], X[:, 1], c=colors, s=15, alpha=0.7, edgecolors="none")

# decision boundary: w^T x = 0  =>  x2 = -(w1/w2) * x1
x1_range = jnp.linspace(float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5, 200)
w_mean = result.mu
x2_boundary = -(w_mean[0] / w_mean[1]) * x1_range
ax.plot(x1_range, x2_boundary, "k-", lw=2, label="Posterior mean boundary")

# true boundary
x2_true = -(TRUE_W[0] / TRUE_W[1]) * x1_range
ax.plot(x1_range, x2_true, "g--", lw=2, label="True boundary")

ax.set_xlim(float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5)
ax.set_ylim(float(X[:, 1].min()) - 0.5, float(X[:, 1].max()) + 0.5)
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Decision boundary")
ax.legend(fontsize=8)

# 3. Posterior weight distributions vs true values
ax = axes[2]
for i, (label, color) in enumerate(zip(["w₁", "w₂"], ["tab:blue", "tab:orange"])):
    mu_i = float(result.mu[i])
    sigma_i = float(result.sigma[i])
    grid = jnp.linspace(mu_i - 4 * sigma_i, mu_i + 4 * sigma_i, 200)
    ax.plot(grid, norm.pdf(grid, mu_i, sigma_i), color=color, lw=2, label=f"q({label})")
    ax.axvline(float(TRUE_W[i]), color=color, ls=":", lw=1.5)
ax.set_xlabel("w")
ax.set_title("Posterior weight distributions")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("bbvi_logistic.png", dpi=150)
print("Saved bbvi_logistic.png")
plt.close(fig)
