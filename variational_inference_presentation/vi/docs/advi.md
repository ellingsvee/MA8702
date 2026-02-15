# Automatic Differentiation Variational Inference (ADVI)

ADVI approximates an intractable posterior $p(\theta \mid y)$ with a
tractable distribution $q(\zeta)$ by maximising the **evidence lower bound**
(ELBO).  Unlike CAVI, which requires model-specific coordinate updates,
ADVI works with *any* differentiable log-density function.

## Variational family

We use a **mean-field Gaussian** in unconstrained space:

$$
q(\zeta) = \mathcal{N}\!\bigl(\mu,\; \operatorname{diag}(\sigma^2)\bigr),
\qquad \sigma_j = \exp(\omega_j).
$$

For our Bayesian linear regression model the unconstrained vector is
$\zeta = (\beta_1, \dots, \beta_p, \log\sigma^2)$.

## ELBO

The ELBO decomposes as

$$
\mathcal{L}(\mu, \omega)
  = \mathbb{E}_{q}\!\bigl[\log p(\zeta \mid y)\bigr]
  + \mathcal{H}[q],
$$

where the entropy of the mean-field Gaussian is

$$
\mathcal{H}[q] = \sum_{j=1}^{d} \omega_j + \frac{d}{2}\log(2\pi e).
$$

## Reparameterisation trick

To obtain low-variance gradients we write

$$
\zeta = \mu + \sigma \odot \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I_d),
$$

so that differentiation passes through $\mu$ and $\omega$ while $\varepsilon$
is treated as fixed noise.

## Optimisation

We maximise $\mathcal{L}$ (equivalently minimise $-\mathcal{L}$) with the
**Adam** optimiser.  Each gradient step uses a small Monte Carlo average
over $S$ reparameterised samples.  The entire loop is compiled with
`jax.lax.scan` for efficient execution on CPU or GPU.
