use crate::filter::{Parameters, Particle, spawn_particles, update_particles};
use crate::maze::Maze;
use crate::robot::{Robot, update_robot};
use crate::window::Item;
use rand::rng;
use raylib::consts::KeyboardKey::*;
use raylib::prelude::*;

/// Weighted mean and 2×2 covariance of the particle cloud.
/// Returns (mean_x, mean_y, var_xx, var_xy, var_yy).
fn particle_stats(particles: &[Particle]) -> Option<(f64, f64, f64, f64, f64)> {
    if particles.is_empty() {
        return None;
    }
    let mx: f64 = particles.iter().map(|p| p.w * p.state.x).sum();
    let my: f64 = particles.iter().map(|p| p.w * p.state.y).sum();
    let vxx: f64 = particles
        .iter()
        .map(|p| p.w * (p.state.x - mx).powi(2))
        .sum();
    let vxy: f64 = particles
        .iter()
        .map(|p| p.w * (p.state.x - mx) * (p.state.y - my))
        .sum();
    let vyy: f64 = particles
        .iter()
        .map(|p| p.w * (p.state.y - my).powi(2))
        .sum();
    Some((mx, my, vxx, vxy, vyy))
}

/// Draw a 2-sigma covariance ellipse and a cross at the mean.
fn draw_estimate(d: &mut RaylibDrawHandle, particles: &[Particle]) {
    let Some((mx, my, vxx, vxy, vyy)) = particle_stats(particles) else {
        return;
    };

    // Eigenvalues of [[vxx, vxy],[vxy, vyy]]
    let trace = vxx + vyy;
    let det = vxx * vyy - vxy * vxy;
    let disc = (trace * trace / 4.0 - det).max(0.0).sqrt();
    let l1 = (trace / 2.0 + disc).max(0.0);
    let l2 = (trace / 2.0 - disc).max(0.0);

    // Rotation angle of the first eigenvector
    let theta = 0.5 * (2.0 * vxy).atan2(vxx - vyy);

    // 2-sigma radii
    let r1 = 2.0 * l1.sqrt();
    let r2 = 2.0 * l2.sqrt();

    let cos_t = theta.cos() as f32;
    let sin_t = theta.sin() as f32;

    // Draw ellipse as a thick polyline
    let n_seg = 60;
    let ellipse_color = Color::new(0x00, 0xFF, 0xFF, 0xFF);
    let thickness = 2.5;
    for i in 0..n_seg {
        let a0 = 2.0 * std::f32::consts::PI * (i as f32) / (n_seg as f32);
        let a1 = 2.0 * std::f32::consts::PI * ((i + 1) as f32) / (n_seg as f32);

        let (ex0, ey0) = ellipse_point(r1 as f32, r2 as f32, cos_t, sin_t, a0);
        let (ex1, ey1) = ellipse_point(r1 as f32, r2 as f32, cos_t, sin_t, a1);

        d.draw_line_ex(
            Vector2::new(mx as f32 + ex0, my as f32 + ey0),
            Vector2::new(mx as f32 + ex1, my as f32 + ey1),
            thickness,
            ellipse_color,
        );
    }

    // Draw mean cross
    let cross = 8.0;
    let mean_color = Color::new(0x00, 0xFF, 0xFF, 0xFF);
    let mc = Vector2::new(mx as f32, my as f32);
    d.draw_line_ex(
        Vector2::new(mc.x - cross, mc.y),
        Vector2::new(mc.x + cross, mc.y),
        thickness,
        mean_color,
    );
    d.draw_line_ex(
        Vector2::new(mc.x, mc.y - cross),
        Vector2::new(mc.x, mc.y + cross),
        thickness,
        mean_color,
    );
}

fn ellipse_point(r1: f32, r2: f32, cos_t: f32, sin_t: f32, angle: f32) -> (f32, f32) {
    let ca = angle.cos();
    let sa = angle.sin();
    let x = r1 * ca;
    let y = r2 * sa;
    (cos_t * x - sin_t * y, sin_t * x + cos_t * y)
}

/// Draw one bar per particle showing its importance weight.
fn draw_weight_bars(d: &mut RaylibDrawHandle, particles: &[Particle]) {
    if particles.is_empty() {
        return;
    }

    let w_max = particles.iter().map(|p| p.w).fold(0.0_f64, f64::max);
    if w_max <= 0.0 {
        return;
    }

    // Layout: top-right corner
    let plot_w: i32 = 200;
    let plot_h: i32 = 90;
    let margin: i32 = 10;
    let x0 = WIDTH - plot_w - margin;
    let y0 = margin;
    let bar_w = (plot_w as f32 / particles.len() as f32).max(1.0);

    // Background
    d.draw_rectangle(
        x0 - 4,
        y0 - 4,
        plot_w + 8,
        plot_h + 8,
        Color::new(0, 0, 0, 0xB0),
    );

    let bar_color = Color::new(0x66, 0xBB, 0xFF, 0xE0);

    for (i, p) in particles.iter().enumerate() {
        let h = (p.w / w_max * plot_h as f64) as i32;
        let bx = x0 + (i as f32 * bar_w) as i32;
        d.draw_rectangle(bx, y0 + plot_h - h, bar_w.ceil() as i32, h, bar_color);
    }
}

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;
const MAZE_COLS: usize = 10;
const MAZE_ROWS: usize = 8;
const DEFAULT_N_PARTICLES: usize = 500;
const SENSOR_NOISE: f64 = 80.0;
const STATE_NOISE: f64 = 10.0;
const C: f64 = 0.5;
const DROPOUT: f64 = 0.1;

pub fn run() {
    let n_particles: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_PARTICLES);
    let (mut rl, thread) = raylib::init()
        .size(WIDTH, HEIGHT)
        .title("Particle Filter")
        // .fullscreen()
        .vsync()
        .build();

    rl.set_target_fps(20);

    let mut show_bars = true;

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
    let mut particles = spawn_particles(n_particles, WIDTH as usize, HEIGHT as usize, &mut rng);

    let parameters = Parameters {
        sigma_xi: STATE_NOISE,
        sigma_epsilon: SENSOR_NOISE,
    };

    while !rl.window_should_close() {
        // Reset
        if rl.is_key_down(KEY_R) {
            let (start_x, start_y) = maze.random_cell_center(&mut rng);
            robot[0] = Robot::new(start_x, start_y);
            particles = spawn_particles(n_particles, WIDTH as usize, HEIGHT as usize, &mut rng);
        }

        // Kidnap: move robot to random location without re-spreading particles
        if rl.is_key_pressed(KEY_K) {
            let (new_x, new_y) = maze.random_cell_center(&mut rng);
            robot[0].x = new_x;
            robot[0].y = new_y;
            robot[0].sensors = maze.sense(new_x, new_y);
        }

        // Toggle weight bars
        if rl.is_key_pressed(KEY_H) {
            show_bars = !show_bars;
        }

        // Re-spread particles without moving the robot
        if rl.is_key_pressed(KEY_S) {
            particles = spawn_particles(n_particles, WIDTH as usize, HEIGHT as usize, &mut rng);
        }

        let (dx, dy) = update_robot(&rl, &mut robot, &maze);

        if dx != 0.0 || dy != 0.0 {
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
        }

        // Render
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(0x10, 0x10, 0x18, 0xFF));

        // Draw maze walls
        for wall in &maze.walls {
            let x = wall.x0 as f32;
            let y = wall.y0 as f32;
            let w = (wall.x1 - wall.x0) as f32;
            let h = (wall.y1 - wall.y0) as f32;
            d.draw_rectangle(x as i32, y as i32, w as i32, h as i32, Color::GRAY);
        }

        // Draw sensor lines
        let r = &robot[0];
        let sensor_color = Color::new(0x44, 0xFF, 0x44, 0xFF);
        let cx = r.x as f32;
        let cy = r.y as f32;

        if let Some(left) = r.sensors.left {
            let x0 = (r.x - left) as f32;
            d.draw_line(x0 as i32, cy as i32, cx as i32, cy as i32, sensor_color);
        }
        if let Some(right) = r.sensors.right {
            let x1 = (r.x + right) as f32;
            d.draw_line(cx as i32, cy as i32, x1 as i32, cy as i32, sensor_color);
        }
        if let Some(up) = r.sensors.up {
            let y0 = (r.y - up) as f32;
            d.draw_line(cx as i32, y0 as i32, cx as i32, cy as i32, sensor_color);
        }
        if let Some(down) = r.sensors.down {
            let y1 = (r.y + down) as f32;
            d.draw_line(cx as i32, cy as i32, cx as i32, y1 as i32, sensor_color);
        }

        // Draw particles
        for p in &particles {
            let half = p.size() as f32;
            d.draw_rectangle(
                (p.state.x - half as f64) as i32,
                (p.state.y - half as f64) as i32,
                (half * 2.0) as i32,
                (half * 2.0) as i32,
                Color::WHITE,
            );
        }

        // Draw robot
        let robot_half = r.size as f32;
        d.draw_rectangle(
            (r.x - r.size) as i32,
            (r.y - r.size) as i32,
            (robot_half * 2.0) as i32,
            (robot_half * 2.0) as i32,
            Color::RED,
        );

        // Draw estimated mean + covariance ellipse
        draw_estimate(&mut d, &particles);

        // Draw weight bars
        if show_bars {
            draw_weight_bars(&mut d, &particles);
        }
    }
}
