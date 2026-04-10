use raylib::consts::KeyboardKey::*;
use raylib::prelude::*;

struct Ball {
    position: Vector2,
    speed: f32,
    radius: f32,
    color: Color,
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("Particle Filter Presentation")
        .vsync()
        .build();

    let mut ball = Ball {
        position: Vector2::new(400.0, 300.0),
        speed: 3.0,
        radius: 40.0,
        color: Color::GREEN,
    };

    while !rl.window_should_close() {
        if rl.is_key_down(KEY_RIGHT) {
            ball.position.x += ball.speed;
        }
        if rl.is_key_down(KEY_LEFT) {
            ball.position.x -= ball.speed;
        }
        if rl.is_key_down(KEY_UP) {
            ball.position.y -= ball.speed;
        }
        if rl.is_key_down(KEY_DOWN) {
            ball.position.y += ball.speed;
        }

        if rl.is_key_pressed(KEY_SPACE) {
            if ball.color == Color::GREEN {
                ball.color = Color::RED;
            } else {
                ball.color = Color::GREEN;
            }
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        d.draw_circle_v(ball.position, ball.radius, ball.color);
    }
}
