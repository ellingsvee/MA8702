from utils import load_sensor_data
from plotting import plot_joint_filter_map
from extended_kalman import extended_kalman_filter
from particle import particle_filter
from ensemble_kalman import ensemble_kalman_filter


def run():
    sensor_data = load_sensor_data()
    states_kalman, covariances_kalman = extended_kalman_filter(sensor_data)
    states_particle, covariances_particle = particle_filter(sensor_data, B=10_000)
    states_ensemble, covariances_ensemble = ensemble_kalman_filter(sensor_data, B=1_000)

    states_particle_B100, covariances_particle_B100 = particle_filter(
        sensor_data, B=100
    )
    states_ensemble_B100, covariances_ensemble_B100 = ensemble_kalman_filter(
        sensor_data, B=100
    )

    plot_joint_filter_map(
        [
            states_kalman,
            states_particle,
            states_particle_B100,
            states_ensemble,
            states_ensemble_B100,
        ],
        [
            covariances_kalman,
            covariances_particle,
            covariances_particle_B100,
            covariances_ensemble,
            covariances_ensemble_B100,
        ],
        [
            "Extended Kalman Filter",
            r"Particle Filter ($B=10000$)",
            r"Particle Filter ($B=100$)",
            r"Ensemble Kalman Filter ($B=1000$)",
            r"Ensemble Kalman Filter ($B=100$)",
        ],
        save_name="joint_filter_map.svg",
    )


if __name__ == "__main__":
    run()
