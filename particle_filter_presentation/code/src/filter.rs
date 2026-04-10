use crate::{
    maze::{Maze, Sensors},
    robot::Robot,
    window::Item,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;

#[derive(Clone, Debug, Copy)]
pub struct State {
    pub x: f64,
    pub y: f64,
}

pub struct Particle {
    pub state: State,
    pub w: f64,
}

impl Item for Particle {
    fn position(&self) -> (f64, f64) {
        (self.state.x, self.state.y)
    }

    fn color(&self) -> u32 {
        0xFFFFFFFF
    }

    fn size(&self) -> f64 {
        6.0
    }
}

pub struct Parameters {
    pub sigma_xi: f64,
    pub sigma_epsilon: f64,
}

fn likelihood(sensors: &Sensors, state: &State, maze: &Maze, sigma_epsilon: f64) -> f64 {
    let var = sigma_epsilon * sigma_epsilon;
    let state_readings = maze.sense(state.x, state.y);

    // Collect squared errors for active sensors only
    let pairs: [(Option<f64>, Option<f64>); 4] = [
        (sensors.left, state_readings.left),
        (sensors.right, state_readings.right),
        (sensors.up, state_readings.up),
        (sensors.down, state_readings.down),
    ];

    let mut sum_sq = 0.0;
    let mut n_active = 0.0;
    for (obs, pred) in &pairs {
        if let (Some(o), Some(p)) = (obs, pred) {
            sum_sq += (o - p).powi(2);
            n_active += 1.0;
        }
    }

    if n_active == 0.0 {
        return 1.0; // No active sensors — uniform weight
    }

    let log_p = -0.5 * n_active * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * n_active * var.ln()
        - 0.5 * (1.0 / var) * sum_sq;
    log_p.exp()
}

fn systematic_resampling(particles: &mut Vec<Particle>, rng: &mut impl Rng) {
    let n = particles.len();
    let n_f = n as f64;
    let u_dist = Uniform::new(0.0, 1.0 / n_f).unwrap();
    let u: f64 = u_dist.sample(rng);

    // Compute cumulative weights
    let mut cumulative = Vec::with_capacity(n);
    let mut cum = 0.0;
    for p in particles.iter() {
        cum += p.w;
        cumulative.push(cum);
    }

    let mut new_particles = Vec::with_capacity(n);
    let mut r = 0;
    for i in 0..n {
        let u_i = (i as f64) / n_f + u;
        while cumulative[r] < u_i {
            r += 1;
        }
        new_particles.push(Particle {
            state: particles[r].state,
            w: 1.0 / n_f,
        });
    }

    *particles = new_particles;
}

fn n_eff(particles: &[Particle]) -> f64 {
    let sum_w_sq: f64 = particles.iter().map(|p| p.w * p.w).sum();
    1.0 / sum_w_sq
}

pub fn spawn_particles(
    n_particles: usize,
    window_width: usize,
    window_height: usize,
    rng: &mut impl Rng,
) -> Vec<Particle> {
    let init_dist_x = Uniform::new(0.0, window_width as f64).unwrap();
    let init_dist_y = Uniform::new(0.0, window_height as f64).unwrap();
    (0..n_particles)
        .map(|_| Particle {
            state: State {
                x: init_dist_x.sample(rng),
                y: init_dist_y.sample(rng),
            },
            w: 1.0 / n_particles as f64,
        })
        .collect()
}

pub fn update_particles(
    particles: &mut Vec<Particle>,
    params: &Parameters,
    robot: &Robot,
    maze: &Maze,
    c: f64,
    robot_dx: f64,
    robot_dy: f64,
    rng: &mut impl Rng,
) {
    if particles.is_empty() {
        return;
    }
    let n_f = particles.len() as f64;

    let sigma_xi = params.sigma_xi;
    let sigma_epsilon = params.sigma_epsilon;

    // Update weights based on the likelihood of the observed sensor readings
    let robot_sensors = robot.get_noisy_sensor_readings(sigma_epsilon, rng);

    // Predict + weight in parallel (each particle gets its own thread-local rng)
    particles.par_iter_mut().for_each(|particle| {
        let mut rng = rand::rng();
        let xi_dist = Normal::new(0.0, sigma_xi).unwrap();

        // Propagate: apply the same movement as the robot, plus noise
        particle.state.x += robot_dx + xi_dist.sample(&mut rng);
        particle.state.y += robot_dy + xi_dist.sample(&mut rng);

        // Update weights (accumulate, don't replace)
        particle.w *= likelihood(&robot_sensors, &particle.state, maze, sigma_epsilon);
    });

    // Normalize weights
    let w_sum: f64 = particles.par_iter().map(|p| p.w).sum();
    particles.par_iter_mut().for_each(|p| p.w /= w_sum);

    // Estimate the estimated state
    // TODO

    // Resample
    if n_eff(&particles) <= c * n_f {
        systematic_resampling(particles, rng);
    }
}
