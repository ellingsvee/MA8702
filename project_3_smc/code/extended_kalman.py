from jax import jit, Array
import jax.numpy as jnp
from utils import initialize, get_matrices, H_matrix, load_sensor_data
from plotting import plot_filter_map, plot_filter_variances


def extended_kalman_filter(sensor_data):
    state, P = initialize()
    A, Q, R = get_matrices()

    @jit
    def predict(state, P) -> tuple[Array, Array]:
        state_pred = A @ state
        P_pred = A @ P @ A.T + Q
        return state_pred, P_pred

    @jit
    def update(state, P, measurement) -> tuple[Array, Array]:
        H, H_jac = H_matrix(state)
        z = measurement - H
        S = H_jac @ P @ H_jac.T + R
        K = P @ H_jac.T @ jnp.linalg.inv(S)
        state_updated = state + K @ z
        P_updated = (jnp.eye(len(state)) - K @ H_jac) @ P
        return state_updated, P_updated

    states = [state]
    covariances = [P]

    for measurement in sensor_data:
        state, P = predict(state, P)
        state, P = update(state, P, measurement)
        states.append(state)
        covariances.append(P)

    states = jnp.stack(states)
    covariances = jnp.stack(covariances)
    return states, covariances


def main():
    sensor_data = load_sensor_data()
    states, covariances = extended_kalman_filter(sensor_data)
    plot_filter_map(states, covariances, title="Extended Kalman Filter")
    plot_filter_variances(states, covariances, title="Extended Kalman Filter")


if __name__ == "__main__":
    main()
