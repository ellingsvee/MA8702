import jax
import jax.random as jr
import jax.numpy as jnp
import pathlib


from spatial import (
    sample_centered_grf,
    estimate_parameters,
    kriging,
    generate_grid,
    cross_mask,
    random_observation_matrix,
    plot_sample,
    plot_ll_history,
    matern,
    mean_function,
)


jax.config.update("jax_enable_x64", True)


PATH = pathlib.Path(__file__).parent
PLT_PATH = PATH / "figures" / "exotic"
PLT_PATH.mkdir(exist_ok=True)


if __name__ == "__main__":
    key = jr.key(0)
    key, sample_key, obs_key, noise_key = jr.split(key, 4)

    # Sample GRF on the full grid
    s_grid = generate_grid(N=100)
    x_grid = sample_centered_grf(
        sample_key,
        s_grid,
        cov_function=lambda h: matern(h, [1.0, 10.0]),
    )

    alpha = 1.0
    x_full_grid = x_grid + mean_function(s_grid) * alpha

    # Extract cross-shaped subset from the grid
    mask = cross_mask(s_grid)
    s = s_grid[mask]
    x_full = x_full_grid[mask]

    plot_sample(s_grid, x_grid, filename=PLT_PATH / "grf_full_field.svg")

    # Subsample observations from the cross
    s_obs, A = random_observation_matrix(obs_key, s, N_obs=200)
    y_obs = A @ x_full + jr.normal(noise_key, (200,)) * 0.05

    plot_sample(s_obs, y_obs, filename=PLT_PATH / "grf_observations.svg")

    (alpha_estim, cov_params_estim, tau_estim), loglikelihood_history = (
        estimate_parameters(
            s_obs,
            y_obs=y_obs,
            init_mean_param=jnp.array([0.5]),
            init_cov_params=jnp.array([1.0, 10.0]),
            init_obs_noise_param=jnp.array([0.05]),
            cov_function=matern,
            mean_function=mean_function,
            max_iter=200,
        )
    )

    print(f"Estimated mean parameter: {alpha_estim:.4f}")
    print(f"Estimated covariance parameters: {cov_params_estim}")
    print(f"Estimated observation noise parameter: {tau_estim:.4f}")

    plot_ll_history(
        loglikelihood_history, filename=PLT_PATH / "grf_loglikelihood_history.svg"
    )

    # Predict on the full grid
    s_pred = generate_grid(N=25)
    mean_pred, var_pred = kriging(
        s_pred=s_pred,
        s_obs=s_obs,
        y_obs=y_obs,
        mean_param=alpha_estim,
        cov_params=cov_params_estim,
        obs_noise_param=tau_estim,
        cov_function=matern,
        mean_function=mean_function,
    )

    plot_sample(
        s_pred,
        mean_pred,
        filename=PLT_PATH / "grf_pred_mean.svg",
        heatmap=True,
        vmin=-3,
        vmax=3,
    )
    plot_sample(
        s_pred,
        var_pred,
        filename=PLT_PATH / "grf_pred_var.svg",
        heatmap=True,
        vmin=0.0,
        vmax=None,
        cmap="inferno",
    )
