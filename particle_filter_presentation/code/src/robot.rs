use crate::maze::{Maze, Sensors};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use raylib::consts::KeyboardKey::*;
use raylib::prelude::*;

pub struct Robot {
    pub x: f64,
    pub y: f64,
    pub v: f64,
    pub size: f64,
    pub color: u32,
    pub sensors: Sensors,
}

impl Robot {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            v: 3.0,
            size: 6.0,
            color: 0xFFFF0000,
            sensors: Sensors {
                left: None,
                right: None,
                up: None,
                down: None,
            },
        }
    }

    pub fn get_noisy_sensor_readings(&self, sensor_noise: f64, rng: &mut impl Rng) -> Sensors {
        let noise_dist = Normal::new(0.0, sensor_noise).unwrap();
        Sensors {
            left: self.sensors.left.map(|v| v + noise_dist.sample(rng)),
            right: self.sensors.right.map(|v| v + noise_dist.sample(rng)),
            up: self.sensors.up.map(|v| v + noise_dist.sample(rng)),
            down: self.sensors.down.map(|v| v + noise_dist.sample(rng)),
        }
    }
}

/// Returns (dx, dy) — the actual displacement applied to the robot.
pub fn update_robot(rl: &RaylibHandle, robot: &mut [Robot], maze: &Maze) -> (f64, f64) {
    let mut x_move = 0.0;
    let mut y_move = 0.0;

    if rl.is_key_down(KEY_LEFT) {
        x_move -= 1.0;
    }
    if rl.is_key_down(KEY_RIGHT) {
        x_move += 1.0;
    }
    if rl.is_key_down(KEY_UP) {
        y_move -= 1.0;
    }
    if rl.is_key_down(KEY_DOWN) {
        y_move += 1.0;
    }

    let size = robot[0].size;
    let old_x = robot[0].x;
    let old_y = robot[0].y;

    // Try x movement first, then y — allows sliding along walls
    let try_x = (old_x + x_move * robot[0].v).clamp(size, maze.width - size);
    let new_x = if !maze.collides(try_x, old_y, size) {
        try_x
    } else {
        old_x
    };

    let try_y = (old_y + y_move * robot[0].v).clamp(size, maze.height - size);
    let new_y = if !maze.collides(new_x, try_y, size) {
        try_y
    } else {
        old_y
    };

    let dx = new_x - old_x;
    let dy = new_y - old_y;

    let all_sensors = maze.sense(new_x, new_y);

    for r in robot {
        r.x = new_x;
        r.y = new_y;
        r.sensors = all_sensors.clone();
    }

    (dx, dy)
}
