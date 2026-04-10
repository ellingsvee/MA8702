from os import PathLike
from pathlib import Path
import numpy as np
from jax import Array, jacrev
import jax.numpy as jnp


def load_sensor_data(folder_path: PathLike = Path("sensor_data")) -> Array:
    sensor_a = np.loadtxt(Path(folder_path) / "sensorA.txt")
    sensor_b = np.loadtxt(Path(folder_path) / "sensorB.txt")
    return jnp.stack([sensor_a, sensor_b], axis=1)  # (T, 2)


def initialize() -> tuple[Array, Array]:
    initial_state = jnp.array([10.0, 30.0, 10.0, -10.0])
    P = jnp.diag(jnp.array([5.0, 5.0, 2.0, 2.0]) ** 2)
    return initial_state, P


def A_matrix(delta: float) -> Array:
    return jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])


def Q_matrix() -> Array:
    return jnp.diag(jnp.array([0.1, 0.1, 0.5, 0.5]) ** 2)


def R_matrix() -> Array:
    return jnp.diag(jnp.array([0.1, 0.1]) ** 2)


def get_matrices() -> tuple[Array, Array, Array]:
    delta = 1.0 / 60.0
    A = A_matrix(delta)
    Q = Q_matrix()
    R = R_matrix()
    return A, Q, R


def h_func(state) -> Array:
    E, N, vx, vy = state
    h = jnp.array([jnp.arctan(E / N), jnp.arctan((40.0 - N) / (40.0 - E))])
    return h


def H_matrix(state) -> tuple[Array, Array]:
    def h_func_aux(state):
        h = h_func(state)
        return h, h

    H_jac, H = jacrev(h_func_aux, has_aux=True)(state)
    return H, H_jac


def initialize_particles() -> tuple[Array, Array]:
    initial_state = jnp.array([10.0, 30.0, 10.0, -10.0])
    P = jnp.diag(jnp.array([5.0, 5.0, 2.0, 2.0]) ** 2)
    return initial_state, P
