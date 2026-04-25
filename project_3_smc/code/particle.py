from jax import jit, lax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from utils import get_matrices, initialize, load_sensor_data, h_func
from plotting import plot_filter_map


def particle_filter(sensor_data, B: int = 10_000, SEED: int = 0):
    key = jr.key(SEED)

    A, Q, R = get_matrices()

    # Intialize particles
    X_init, P0 = initialize()
    key, subkey = jr.split(key)
    particles_init = jr.multivariate_normal(subkey, X_init, P0, shape=(B,))  # (B, 4)
    weights_init = jnp.ones(B) / B

    states = []
    covariances = []

    @jit
    def step_fn(carry, measurement):
        particles, weights, loop_key = carry

        # Propagate
        loop_key, predict_key, update_key = jr.split(loop_key, 3)
        means = jnp.einsum("ij,bj->bi", A, particles)  # (B, 4)
        noise = jr.multivariate_normal(predict_key, jnp.zeros(4), Q, shape=(B,))
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
        # states.append(mean)
        # covariances.append(cov)

        # Resample (currently to this at every time step)
        cumsum = jnp.cumsum(weights)
        u = (jr.uniform(update_key) + jnp.arange(B)) / B
        indices = jnp.searchsorted(cumsum, u)
        particles = particles[indices]
        weights = jnp.ones(B) / B

        return (particles, weights, loop_key), (mean, cov)

    # Iterate over sensor data
    _, (states, covariances) = lax.scan(
        step_fn, (particles_init, weights_init, key), sensor_data
    )

    return states, covariances


def main():
    sensor_data = load_sensor_data()
    states, covariances = particle_filter(sensor_data)
    plot_filter_map(states, covariances, save_name="particle_filter_map.svg")

    states, covariances = particle_filter(sensor_data, B=100)
    plot_filter_map(states, covariances, save_name="particle_filter_map_B100.svg")


if __name__ == "__main__":
    main()
