from jax import jit, lax
import jax.numpy as jnp
import jax.random as jr

from plotting import plot_filter_map
from utils import get_matrices, h_func, initialize, load_sensor_data


def ensemble_kalman_filter(sensor_data, B: int = 1_000, SEED: int = 0):
    key = jr.key(SEED)
    A, Q, R = get_matrices()
    initial_state, P0 = initialize()

    # Initial Ensemble
    key, subkey = jr.split(key)
    X_init = jr.multivariate_normal(subkey, initial_state, P0, shape=(B,))

    @jit
    def step_fn(carry, measurement):
        X, loop_key = carry

        loop_key, predict_key, update_key = jr.split(loop_key, 3)

        q_noise = jr.multivariate_normal(
            predict_key, jnp.zeros(Q.shape[0]), Q, shape=(B,)
        )
        X_predict = jnp.einsum("ij,bj->bi", A, X) + q_noise

        # Project to measurement space
        HX = jnp.vectorize(h_func, signature="(4)->(2)")(X_predict)

        # Perturb measurements
        v_noise = jr.multivariate_normal(
            update_key, jnp.zeros(R.shape[0]), R, shape=(B,)
        )
        Y_perturbed = measurement + v_noise  # shape (B, 2)

        # Calculate empirical covariances
        X_mean = jnp.mean(X_predict, axis=0)
        HX_mean = jnp.mean(HX, axis=0)
        X_centered = X_predict - X_mean
        HX_centered = HX - HX_mean
        Sigma_yy = (HX_centered.T @ HX_centered) / (B - 1) + R
        Sigma_xy = (X_centered.T @ HX_centered) / (B - 1)

        # Update using Kalman gain
        K = jnp.linalg.solve(Sigma_yy, Sigma_xy.T).T
        X_update = X_predict + jnp.einsum("ij,bj->bi", K, Y_perturbed - HX)

        curr_mean = jnp.mean(X_update, axis=0)
        curr_cov = jnp.cov(X_update, rowvar=False)
        return (X_update, loop_key), (curr_mean, curr_cov)

    # Iterate over sensor data
    _, (states, covariances) = lax.scan(step_fn, (X_init, key), sensor_data)

    return states, covariances


def main():
    sensor_data = load_sensor_data()
    states, covariances = ensemble_kalman_filter(sensor_data)
    plot_filter_map(states, covariances, save_name="enkf_B1000_stochastic_map.svg")

    states, covariances = ensemble_kalman_filter(sensor_data, B=100)
    plot_filter_map(states, covariances, save_name="enkf_B100_stochastic_map.svg")


if __name__ == "__main__":
    main()
