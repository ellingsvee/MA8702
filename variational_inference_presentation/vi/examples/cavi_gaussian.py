"""CAVI example: infer the mean and precision of a Gaussian from synthetic data."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm

from vi.cavi import NormalGammaPrior, run_cavi

# ---------------------------------------------------------------------------
# Ground truth and data generation
# ---------------------------------------------------------------------------
TRUE_MU = 3.0
TRUE_TAU = 2.0  # precision (1/variance)
N = 50

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)
x = TRUE_MU + jax.random.normal(key, shape=(N,)) / jnp.sqrt(TRUE_TAU)

# ---------------------------------------------------------------------------
# Run CAVI with a weak prior
# ---------------------------------------------------------------------------
prior = NormalGammaPrior(mu_0=0.0, kappa_0=0.1, a_0=1.0, b_0=1.0)
posterior, elbos = run_cavi(x, prior, n_iter=50)

print("=== CAVI results ===")
print(f"  True mu  = {TRUE_MU:.3f},   Posterior mean of mu  = {float(posterior.m_q):.3f}")
print(f"  True tau = {TRUE_TAU:.3f},   Posterior E[tau]      = {float(posterior.a_q / posterior.b_q):.3f}")
print(f"  Posterior std of mu = {1 / jnp.sqrt(posterior.lambda_q):.4f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. ELBO convergence
ax = axes[0]
ax.plot(jnp.arange(1, len(elbos) + 1), elbos, "o-", markersize=3)
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
ax.set_title("ELBO convergence")

# 2. Data histogram + inferred Gaussian
ax = axes[1]
ax.hist(x, bins=15, density=True, alpha=0.5, label="Data")
xs = jnp.linspace(float(jnp.min(x)) - 1, float(jnp.max(x)) + 1, 300)
E_tau = posterior.a_q / posterior.b_q
inferred_std = 1.0 / jnp.sqrt(E_tau)
ax.plot(xs, norm.pdf(xs, posterior.m_q, inferred_std), "r-", lw=2, label="Inferred N(m_q, 1/E[τ])")
ax.plot(xs, norm.pdf(xs, TRUE_MU, 1.0 / jnp.sqrt(TRUE_TAU)), "k--", lw=2, label="True N(μ, 1/τ)")
ax.set_xlabel("x")
ax.set_title("Data & inferred distribution")
ax.legend(fontsize=8)

# 3. Prior vs posterior for mu
ax = axes[2]
prior_std_mu = 1.0 / jnp.sqrt(prior.kappa_0 * (prior.a_0 / prior.b_0))
post_std_mu = 1.0 / jnp.sqrt(posterior.lambda_q)
mu_grid = jnp.linspace(float(prior.mu_0 - 4 * prior_std_mu), float(prior.mu_0 + 4 * prior_std_mu), 300)
ax.plot(mu_grid, norm.pdf(mu_grid, prior.mu_0, prior_std_mu), "b--", lw=2, label="Prior q₀(μ)")
mu_grid2 = jnp.linspace(float(posterior.m_q - 4 * post_std_mu), float(posterior.m_q + 4 * post_std_mu), 300)
ax.plot(mu_grid2, norm.pdf(mu_grid2, posterior.m_q, post_std_mu), "r-", lw=2, label="Posterior q(μ)")
ax.axvline(TRUE_MU, color="k", ls=":", label="True μ")
ax.set_xlabel("μ")
ax.set_title("Prior vs posterior for μ")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("cavi_gaussian.png", dpi=150)
print("Saved cavi_gaussian.png")
plt.close(fig)
