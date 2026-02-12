import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from vi.bbvi import run_bbvi

SEED = 123


# ── Data ──────────────────────────────────────────────────────────────────────


def polynomial_features(inputs):
    x1, x2 = inputs[:, 0:1], inputs[:, 1:2]
    return jnp.concatenate([x1, x2, x1**2, x2**2, x1 * x2], axis=1)


raw_inputs, outputs = make_moons(n_samples=300, noise=0.2, random_state=SEED)
raw_inputs = jnp.array(raw_inputs, dtype=jnp.float32)
outputs = jnp.array(outputs, dtype=jnp.float32)

inputs = polynomial_features(raw_inputs)
# Prepend a column of ones for the bias term
inputs = jnp.concatenate([jnp.ones((inputs.shape[0], 1)), inputs], axis=1)

num_params = inputs.shape[1]  # 6 (bias + 5 polynomial features)


# ── Log-joint for Bayesian logistic regression ────────────────────────────────

PRIOR_STD = 1.0


def log_joint(w):
    """log p(y | X, w) + log p(w)"""
    logits = inputs @ w
    # log p(y | X, w) — Bernoulli log-likelihood
    log_lik = jnp.sum(
        outputs * jax.nn.log_sigmoid(logits)
        + (1 - outputs) * jax.nn.log_sigmoid(-logits)
    )
    # log p(w) — Gaussian prior
    log_prior = -0.5 * jnp.sum(w**2) / PRIOR_STD**2
    return log_lik + log_prior


# ── Run BBVI ──────────────────────────────────────────────────────────────────

mu, sigma = run_bbvi(
    key=jax.random.key(SEED),
    log_joint_fn=log_joint,
    shape=(num_params,),
    n_iter=2000,
    n_samples=50,
    lr=0.05,
)

print("Posterior mean:", mu)
print("Posterior std: ", sigma)


# ── Predictions on a grid ─────────────────────────────────────────────────────

GRID_SIZE = 150
BUFFER = 0.5
NUM_PRED_SAMPLES = 200

x1_min, x1_max = raw_inputs[:, 0].min() - BUFFER, raw_inputs[:, 0].max() + BUFFER
x2_min, x2_max = raw_inputs[:, 1].min() - BUFFER, raw_inputs[:, 1].max() + BUFFER

x1_grid = jnp.linspace(x1_min, x1_max, GRID_SIZE)
x2_grid = jnp.linspace(x2_min, x2_max, GRID_SIZE)
X1_mesh, X2_mesh = jnp.meshgrid(x1_grid, x2_grid, indexing="ij")

grid = jnp.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=1)
grid_features = polynomial_features(grid)
grid_features = jnp.concatenate(
    [jnp.ones((grid_features.shape[0], 1)), grid_features], axis=1
)

# Sample weights from the variational posterior
key = jax.random.key(0)
eps = jax.random.normal(key, shape=(NUM_PRED_SAMPLES, num_params))
weight_samples = mu + sigma * eps  # (NUM_PRED_SAMPLES, num_params)

# Compute predicted probabilities for each weight sample
logits_grid = grid_features @ weight_samples.T  # (grid_points, NUM_PRED_SAMPLES)
probs_grid = jax.nn.sigmoid(logits_grid)

mean_prob = probs_grid.mean(axis=1).reshape(GRID_SIZE, GRID_SIZE)
std_prob = probs_grid.std(axis=1).reshape(GRID_SIZE, GRID_SIZE)


# ── Visualisation ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: mean predictive probability
ax1 = axes[0]
c1 = ax1.contourf(X1_mesh, X2_mesh, mean_prob, levels=20, cmap="RdYlBu_r", alpha=0.8)
ax1.contour(X1_mesh, X2_mesh, mean_prob, levels=[0.5], colors="black", linewidths=2)
ax1.scatter(
    raw_inputs[outputs == 0, 0],
    raw_inputs[outputs == 0, 1],
    s=25,
    c="blue",
    edgecolors="darkblue",
    alpha=0.7,
    label="Class 0",
)
ax1.scatter(
    raw_inputs[outputs == 1, 0],
    raw_inputs[outputs == 1, 1],
    s=25,
    c="red",
    edgecolors="darkred",
    alpha=0.7,
    label="Class 1",
)
plt.colorbar(c1, ax=ax1, label="P(y=1|x)")
ax1.set_title("Mean Predictive Probability", fontsize=12, fontweight="bold")
ax1.set_xlabel("$x_1$", fontsize=11)
ax1.set_ylabel("$x_2$", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# Right: predictive uncertainty
ax2 = axes[1]
c2 = ax2.contourf(X1_mesh, X2_mesh, std_prob, levels=20, cmap="viridis", alpha=0.85)
ax2.scatter(
    raw_inputs[outputs == 0, 0],
    raw_inputs[outputs == 0, 1],
    s=25,
    c="lightblue",
    edgecolors="darkblue",
    alpha=0.6,
)
ax2.scatter(
    raw_inputs[outputs == 1, 0],
    raw_inputs[outputs == 1, 1],
    s=25,
    c="lightcoral",
    edgecolors="darkred",
    alpha=0.6,
)
plt.colorbar(c2, ax=ax2, label="Std(p)")
ax2.set_title("Predictive Uncertainty (Epistemic)", fontsize=12, fontweight="bold")
ax2.set_xlabel("$x_1$", fontsize=11)
ax2.set_ylabel("$x_2$", fontsize=11)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("bayesian_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.close()
