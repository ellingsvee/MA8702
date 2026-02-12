import jax.numpy as jnp
from tabulate import tabulate


def order(centers, parameters):
    m, s2 = parameters[0], parameters[1]

    order = jnp.argsort(m[:, 0])
    m_sorted = m[order]
    s2_sorted = s2[order]
    centers_sorted = centers[jnp.argsort(centers[:, 0])]

    return centers_sorted, m_sorted, s2_sorted


def tabulate_estimates(centers, params, headers=None):
    K, dim = centers.shape

    centers_sorted, m_sorted, s2_sorted = order(centers, params)

    if s2_sorted.ndim == 1:
        s2_sorted = s2_sorted[:, None]

    table = jnp.concatenate(
        [
            jnp.arange(K)[:, None],
            centers_sorted,
            m_sorted,
            s2_sorted,
        ],
        axis=1,
    )

    headers = (
        ["Cluster"]
        + [f"True μ[{d}]" for d in range(dim)]
        + [f"Est μ[{d}]" for d in range(dim)]
        + [f"Est s2[{d}]" for d in range(dim)]
    )

    print(
        tabulate(table.tolist(), headers=headers, tablefmt="fancy_grid", floatfmt=".3f")
    )
