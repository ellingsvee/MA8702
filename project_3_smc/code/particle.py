import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from utils import get_matrices, initialize, load_sensor_data, h_func
from plotting import plot_filter_map, plot_filter_variances


def particle_filter(sensor_data, B: int = 10_000, SEED: int = 0):
    key = jr.key(SEED)

    A, Q, R = get_matrices()

    # Intialize particles
    initial_state, P0 = initialize()
    key, subkey = jr.split(key)
    particles = jr.multivariate_normal(subkey, initial_state, P0, shape=(B,))  # (B, 4)
    weights = jnp.ones(B) / B

    states = []
    covariances = []

    # Initial estimates
    mean = jnp.average(particles, axis=0, weights=weights)
    diff = particles - mean
    cov = jnp.einsum("i,ij,ik->jk", weights, diff, diff)
    states.append(mean)
    covariances.append(cov)

    for measurement in sensor_data:
        # Propagate
        key, subkey = jr.split(key)
        means = jnp.einsum("ij,bj->bi", A, particles)  # (B, 4)
        noise = jr.multivariate_normal(subkey, jnp.zeros(4), Q, shape=(B,))
        particles = means + noise

        # Reweight
        h_vals = jnp.vectorize(h_func, signature="(4)->(2)")(particles)  # (B, 2)
        log_weights = multivariate_normal.logpdf(measurement, mean=h_vals, cov=R)

        # Normalize weights (log-sum-exp trick)
        log_weights = log_weights - jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weights = weights / jnp.sum(weights)

        # Estimate
        mean = jnp.average(particles, axis=0, weights=weights)
        diff = particles - mean
        cov = jnp.einsum("i,ij,ik->jk", weights, diff, diff)
        states.append(mean)
        covariances.append(cov)

        # Resample (currently to this at every time step)
        key, subkey = jr.split(key)
        cumsum = jnp.cumsum(weights)
        u = (jr.uniform(subkey) + jnp.arange(B)) / B
        indices = jnp.searchsorted(cumsum, u)
        particles = particles[indices]
        weights = jnp.ones(B) / B

    states = jnp.stack(states)
    covariances = jnp.stack(covariances)
    return states, covariances


def main():
    sensor_data = load_sensor_data()
    states, covariances = particle_filter(sensor_data)
    plot_filter_map(states, covariances, save_name="particle_filter_map.svg")
    plot_filter_variances(
        states, covariances, save_name="particle_filter_variances.svg"
    )


if __name__ == "__main__":
    main()
