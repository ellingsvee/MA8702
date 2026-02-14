import jax.numpy as jnp


def autocorr(x, max_lag=100, normalize=True):
    x = jnp.asarray(x)
    x = x - jnp.mean(x)

    n = x.shape[0]

    if max_lag is None:
        max_lag = n - 1
    max_lag = jnp.minimum(max_lag, n - 1)

    # FFT with zero-padding
    f = jnp.fft.fft(x, n=2 * n)
    ac = jnp.fft.ifft(f * jnp.conj(f)).real

    # keep only requested lags
    ac = ac[: max_lag + 1]

    if normalize:
        ac = ac / ac[0]

    return ac
