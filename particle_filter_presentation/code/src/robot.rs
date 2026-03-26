use crate::maze::{Maze, Sensors};
use crate::window::Item;
use minifb::{Key, Window};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

pub struct Robot {
    pub x: f64,
    pub y: f64,
    pub v: f64,
    pub size: f64,
    pub color: u32,
    pub sensors: Sensors,
}

impl Item for Robot {
    fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn color(&self) -> u32 {
        self.color
    }

    fn size(&self) -> f64 {
        self.size
    }
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
pub fn update_robot(window: &Window, robot: &mut [Robot], maze: &Maze) -> (f64, f64) {
    let mut x_move = 0.0;
    let mut y_move = 0.0;

    if window.is_key_down(Key::Left) {
        x_move -= 1.0;
    }
    if window.is_key_down(Key::Right) {
        x_move += 1.0;
    }
    if window.is_key_down(Key::Up) {
        y_move -= 1.0;
    }
    if window.is_key_down(Key::Down) {
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

    // Only activate sensors in the direction(s) of movement.
    let sensors = Sensors {
        left: if dx < 0.0 { all_sensors.left } else { None },
        right: if dx > 0.0 { all_sensors.right } else { None },
        up: if dy < 0.0 { all_sensors.up } else { None },
        down: if dy > 0.0 { all_sensors.down } else { None },
    };

    for r in robot {
        r.x = new_x;
        r.y = new_y;
        r.sensors = sensors.clone();
    }

    (dx, dy)
}
