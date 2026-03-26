use crate::filter::{Parameters, Particle, State, spawn_particles, update_particles};
use crate::maze::{Maze, Sensors};
use crate::robot::{Robot, update_robot};
use crate::window::Frame;
use minifb::{Key, Window, WindowOptions};
use rand::rng;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const MAZE_COLS: usize = 10;
const MAZE_ROWS: usize = 8;
const N_PARTICLES: usize = 50;
const SENSOR_NOISE: f64 = 20.0;
const STATE_NOISE: f64 = 10.0;
const C: f64 = 1.0;
const DROPOUT: f64 = 0.6;

pub fn run() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Particle Filter",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");
    window.set_target_fps(20);

    let mut rng = rng();

    let maze = Maze::generate(
        WIDTH as f64,
        HEIGHT as f64,
        MAZE_COLS,
        MAZE_ROWS,
        DROPOUT,
        &mut rng,
    );
    let (start_x, start_y) = maze.random_cell_center(&mut rng);

    let mut robot = vec![Robot::new(start_x, start_y)];

    let mut particles = spawn_particles(N_PARTICLES, WIDTH, HEIGHT, &mut rng);

    let parameters = Parameters {
        sigma_xi: STATE_NOISE,
        sigma_epsilon: SENSOR_NOISE,
    };

    let mut frame = Frame::new(WIDTH, HEIGHT);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Reset
        if window.is_key_down(Key::R) {
            let (start_x, start_y) = maze.random_cell_center(&mut rng);
            robot[0] = Robot::new(start_x, start_y);
            particles = spawn_particles(N_PARTICLES, WIDTH, HEIGHT, &mut rng);
        }

        let (dx, dy) = update_robot(&window, &mut robot, &maze);
        update_particles(
            &mut particles,
            &parameters,
            &robot[0],
            &maze,
            C,
            dx,
            dy,
            &mut rng,
        );

        // render
        buffer.fill(0xFF101018);
        frame.draw_maze(&mut buffer, &maze, 0xFF888888);

        let r = &robot[0];
        frame.draw_sensors(
            &mut buffer,
            r.x,
            r.y,
            r.sensors.left,
            r.sensors.right,
            r.sensors.up,
            r.sensors.down,
            0xFF44FF44,
        );

        frame.clear();
        frame.draw_items(&particles);
        frame.draw_items(&robot);
        frame.draw_frame(&mut buffer);

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .expect("failed to update buffer");
    }
}
