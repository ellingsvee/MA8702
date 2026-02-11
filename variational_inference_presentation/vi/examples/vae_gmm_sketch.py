"""VAE sketch: amortised variational inference on synthetic "image" data.

Key pedagogical point:
    The same reparameterisation trick from BBVI, but now q(z|x) is
    parameterised by a neural network (amortised inference) instead of
    per-sample variational parameters.

Model:
    p(z) = N(0, I)               latent (2D for visualisation)
    p(x|z) = N(decoder(z), I)    observed "images" (8x8 = 64 pixels)

Variational family:
    q(z|x) = N(mu(x), diag(sigma(x)^2))   encoder outputs mu, log_sigma

Uses only JAX (no equinox/flax) — MLPs are defined manually.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Synthetic data: 8x8 "images" of Gaussian blobs at K cluster positions
# ---------------------------------------------------------------------------
IMG_SIZE = 8
IMG_DIM = IMG_SIZE**2  # 64
LATENT_DIM = 2
K = 3  # number of blob positions
N = 500
HIDDEN = 64

# Blob centres in pixel coordinates
BLOB_CENTRES = np.array([[2.0, 2.0], [5.0, 2.0], [3.5, 5.5]])

key = jax.random.key(42)


def make_blob_image(centre, size=IMG_SIZE):
    """Generate an 8x8 image with a Gaussian blob at `centre`."""
    y, x_grid = np.mgrid[0:size, 0:size]
    img = np.exp(-0.5 * ((x_grid - centre[0])**2 + (y - centre[1])**2) / 0.8**2)
    return img.ravel().astype(np.float64)


# Generate dataset
key, k1, k2 = jax.random.split(key, 3)
labels = jax.random.categorical(k1, jnp.zeros(K), shape=(N,))
images = []
for i in range(N):
    c = BLOB_CENTRES[int(labels[i])]
    # Add small jitter to blob position
    jitter = np.array(jax.random.normal(k2, shape=(2,))) * 0.3
    k2, _ = jax.random.split(k2)
    img = make_blob_image(c + jitter)
    images.append(img)
x_data = jnp.array(np.stack(images))  # (N, 64)

print(f"Data: {N} images of size {IMG_SIZE}x{IMG_SIZE}, K={K} clusters, latent_dim={LATENT_DIM}")

# ---------------------------------------------------------------------------
# MLP utilities (pure JAX, no frameworks)
# ---------------------------------------------------------------------------


def init_mlp(key, layer_sizes):
    """Initialise an MLP with Xavier initialisation. Returns list of (W, b)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        W = jax.random.normal(subkey, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def mlp_forward(params, x):
    """Forward pass through MLP with ReLU activations (no activation on last layer)."""
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# ---------------------------------------------------------------------------
# Encoder: x -> (mu_z, log_sigma_z)
# Decoder: z -> reconstructed x
# ---------------------------------------------------------------------------


def encode(enc_params, x):
    """Return (mu, log_sigma) for the latent z."""
    h = mlp_forward(enc_params, x)
    mu, log_sigma = h[:LATENT_DIM], h[LATENT_DIM:]
    return mu, log_sigma


def decode(dec_params, z):
    """Return reconstructed x (mean of p(x|z))."""
    return mlp_forward(dec_params, z)


# ---------------------------------------------------------------------------
# VAE loss = -ELBO = reconstruction loss + KL divergence
# ---------------------------------------------------------------------------


def vae_loss(params, x_batch, key):
    """Compute -ELBO averaged over a mini-batch.

    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

    The reparameterisation trick: z = mu + sigma * epsilon, epsilon ~ N(0, I)
    is exactly the same trick used in BBVI, but here mu and sigma are outputs
    of a neural network (amortised over all data points).
    """
    enc_params, dec_params = params
    batch_size = x_batch.shape[0]

    def per_example(x_i, subkey):
        mu_z, log_sigma_z = encode(enc_params, x_i)
        sigma_z = jnp.exp(log_sigma_z)

        # Reparameterisation trick (same as BBVI!)
        eps = jax.random.normal(subkey, shape=(LATENT_DIM,))
        z = mu_z + sigma_z * eps

        # Reconstruction: log p(x|z) under unit-variance Gaussian
        x_recon = decode(dec_params, z)
        recon_loss = -0.5 * jnp.sum((x_i - x_recon) ** 2)

        # KL divergence: KL(N(mu, sigma^2) || N(0, I))
        # = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        kl = 0.5 * jnp.sum(sigma_z**2 + mu_z**2 - 1 - 2 * log_sigma_z)

        return recon_loss - kl  # ELBO for this example

    keys = jax.random.split(key, batch_size)
    elbos = jax.vmap(per_example)(x_batch, keys)
    return -jnp.mean(elbos)  # negative ELBO (to minimise)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
import optax

# Initialise networks
key, k_enc, k_dec = jax.random.split(key, 3)
enc_params = init_mlp(k_enc, [IMG_DIM, HIDDEN, HIDDEN, 2 * LATENT_DIM])
dec_params = init_mlp(k_dec, [LATENT_DIM, HIDDEN, HIDDEN, IMG_DIM])
params = (enc_params, dec_params)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

grad_fn = jax.jit(jax.value_and_grad(vae_loss))

N_EPOCHS = 80
BATCH_SIZE = 64
elbos_trace = []

print("Training VAE...")
for epoch in range(N_EPOCHS):
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, N, BATCH_SIZE):
        batch = x_data[perm[start : start + BATCH_SIZE]]
        key, subkey = jax.random.split(key)
        loss, grads = grad_fn(params, batch, subkey)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        epoch_loss += float(loss)
        n_batches += 1

    avg_elbo = -epoch_loss / n_batches
    elbos_trace.append(avg_elbo)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: ELBO ~ {avg_elbo:.1f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
enc_params_final, dec_params_final = params

# Encode all data into latent space
all_mu = []
for i in range(N):
    mu_z, _ = encode(enc_params_final, x_data[i])
    all_mu.append(mu_z)
latents = jnp.stack(all_mu)  # (N, 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. ELBO convergence
ax = axes[0]
ax.plot(range(1, N_EPOCHS + 1), elbos_trace, "o-", markersize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("ELBO")
ax.set_title("VAE ELBO convergence")

# 2. Latent space coloured by true cluster
ax = axes[1]
colors = ["tab:blue", "tab:orange", "tab:green"]
for k in range(K):
    mask = np.array(labels) == k
    ax.scatter(
        np.array(latents[mask, 0]), np.array(latents[mask, 1]),
        c=colors[k], alpha=0.5, s=20, label=f"Cluster {k+1}",
    )
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("Latent space (coloured by true cluster)")
ax.legend(fontsize=8)
ax.set_aspect("equal")

# 3. Reconstructions vs originals (show 8 examples)
ax = axes[2]
n_show = 8
key, subkey = jax.random.split(key)
show_idx = jax.random.choice(subkey, N, shape=(n_show,), replace=False)

# Build a grid: top row = original, bottom row = reconstruction
grid = np.zeros((2 * IMG_SIZE, n_show * IMG_SIZE))
for j, idx in enumerate(show_idx):
    orig = np.array(x_data[int(idx)]).reshape(IMG_SIZE, IMG_SIZE)
    mu_z, _ = encode(enc_params_final, x_data[int(idx)])
    recon = np.array(decode(dec_params_final, mu_z)).reshape(IMG_SIZE, IMG_SIZE)
    grid[:IMG_SIZE, j * IMG_SIZE : (j + 1) * IMG_SIZE] = orig
    grid[IMG_SIZE:, j * IMG_SIZE : (j + 1) * IMG_SIZE] = recon

ax.imshow(grid, cmap="viridis", aspect="auto")
ax.set_title("Originals (top) vs Reconstructions (bottom)")
ax.set_yticks([IMG_SIZE // 2, IMG_SIZE + IMG_SIZE // 2])
ax.set_yticklabels(["Original", "Recon"])
ax.set_xticks([])

fig.tight_layout()
fig.savefig("output/vae_gmm_sketch.png", dpi=150)
print("Saved output/vae_gmm_sketch.png")
plt.close(fig)
