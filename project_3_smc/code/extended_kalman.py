from jax import jit, Array, lax
import jax.numpy as jnp
from utils import initialize, get_matrices, H_matrix, load_sensor_data
from plotting import plot_filter_map


def extended_kalman_filter(sensor_data):
    state_init, P_init = initialize()
    A, Q, R = get_matrices()

    def predict(state, P) -> tuple[Array, Array]:
        state_pred = A @ state
        P_pred = A @ P @ A.T + Q
        return state_pred, P_pred

    def update(state, P, measurement) -> tuple[Array, Array]:
        H, H_jac = H_matrix(state)
        z = measurement - H
        S = H_jac @ P @ H_jac.T + R
        K = P @ H_jac.T @ jnp.linalg.inv(S)
        state_updated = state + K @ z
        P_updated = (jnp.eye(len(state)) - K @ H_jac) @ P
        return state_updated, P_updated

    @jit
    def step(state, P, measurement):
        state_pred, P_pred = predict(state, P)
        state_updated, P_updated = update(state_pred, P_pred, measurement)
        return state_updated, P_updated

    @jit
    def step_fn(carry, measurement):
        state, P = carry
        state_pred = A @ state
        P_pred = A @ P @ A.T + Q

        H, H_jac = H_matrix(state_pred)
        z = measurement - H
        S = H_jac @ P_pred @ H_jac.T + R
        K = P_pred @ H_jac.T @ jnp.linalg.inv(S)
        state_updated = state_pred + K @ z
        P_updated = (jnp.eye(len(state_pred)) - K @ H_jac) @ P_pred

        return (state_updated, P_updated), (state_updated, P_updated)

    # Iterate over sensor data
    _, (states, covariances) = lax.scan(step_fn, (state_init, P_init), sensor_data)
    return states, covariances


def main():
    sensor_data = load_sensor_data()
    states, covariances = extended_kalman_filter(sensor_data)
    plot_filter_map(states, covariances, save_name="extended_kalman_filter_map.svg")


if __name__ == "__main__":
    main()
