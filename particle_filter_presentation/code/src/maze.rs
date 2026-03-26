use rand::Rng;
use rand::prelude::IndexedRandom;
use rand_distr::{Distribution, Uniform};

const WALL_THICKNESS: f64 = 4.0;

pub struct Wall {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

#[derive(Clone, Debug)]
pub struct Sensors {
    pub left: Option<f64>,
    pub right: Option<f64>,
    pub up: Option<f64>,
    pub down: Option<f64>,
}

pub struct Maze {
    pub walls: Vec<Wall>,
    pub width: f64,
    pub height: f64,
    pub cell_w: f64,
    pub cell_h: f64,
}

impl Maze {
    /// Generate a maze using recursive backtracking.
    /// `cols` x `rows` cells filling the entire `width` x `height` domain.
    pub fn generate(
        width: f64,
        height: f64,
        cols: usize,
        rows: usize,
        dropout: f64,
        rng: &mut impl Rng,
    ) -> Self {
        // let mut rng = rand::rng();
        let cell_w = width / cols as f64;
        let cell_h = height / rows as f64;
        let half_t = WALL_THICKNESS / 2.0;

        // h_walls[row][col]: horizontal wall on top edge of cell (row, col)
        let mut h_walls = vec![vec![true; cols]; rows + 1];
        // v_walls[row][col]: vertical wall on left edge of cell (row, col)
        let mut v_walls = vec![vec![true; cols + 1]; rows];

        let mut visited = vec![vec![false; cols]; rows];
        let mut stack: Vec<(usize, usize)> = Vec::new();

        visited[0][0] = true;
        stack.push((0, 0));

        while let Some(&(r, c)) = stack.last() {
            let mut neighbors = Vec::new();
            if r > 0 && !visited[r - 1][c] {
                neighbors.push((r - 1, c));
            }
            if r + 1 < rows && !visited[r + 1][c] {
                neighbors.push((r + 1, c));
            }
            if c > 0 && !visited[r][c - 1] {
                neighbors.push((r, c - 1));
            }
            if c + 1 < cols && !visited[r][c + 1] {
                neighbors.push((r, c + 1));
            }

            if neighbors.is_empty() {
                stack.pop();
            } else {
                let &(nr, nc) = neighbors.choose(rng).unwrap();
                if nr < r {
                    h_walls[r][c] = false;
                } else if nr > r {
                    h_walls[nr][c] = false;
                } else if nc < c {
                    v_walls[r][c] = false;
                } else {
                    v_walls[r][nc] = false;
                }
                visited[nr][nc] = true;
                stack.push((nr, nc));
            }
        }

        let mut walls = Vec::new();

        let dropout_dist = Uniform::new(0.0, 1.0).unwrap();
        let mut u;

        // Horizontal wall segments
        for row in 0..=rows {
            let y = row as f64 * cell_h;
            for col in 0..cols {
                if h_walls[row][col] {
                    let x0 = col as f64 * cell_w;
                    let x1 = (col + 1) as f64 * cell_w;

                    u = dropout_dist.sample(rng);
                    if row > 0 && row < rows && u < dropout {
                        continue;
                    }

                    let wall = Wall {
                        x0: x0 - half_t,
                        y0: y - half_t,
                        x1: x1 + half_t,
                        y1: y + half_t,
                    };
                    walls.push(wall);
                }
            }
        }

        // Vertical wall segments
        for row in 0..rows {
            for col in 0..=cols {
                if v_walls[row][col] {
                    let x = col as f64 * cell_w;
                    let y0 = row as f64 * cell_h;
                    let y1 = (row + 1) as f64 * cell_h;

                    u = dropout_dist.sample(rng);
                    if col > 0 && col < cols && u < dropout {
                        continue;
                    }

                    let wall = Wall {
                        x0: x - half_t,
                        y0: y0 - half_t,
                        x1: x + half_t,
                        y1: y1 + half_t,
                    };
                    walls.push(wall);
                }
            }
        }

        Maze {
            walls,
            width,
            height,
            cell_w,
            cell_h,
        }
    }

    /// Returns the center of a given cell — useful for spawning.
    pub fn cell_center(&self, col: usize, row: usize) -> (f64, f64) {
        (
            (col as f64 + 0.5) * self.cell_w,
            (row as f64 + 0.5) * self.cell_h,
        )
    }

    /// Check if a square (center cx,cy with half-width h) overlaps any wall.
    pub fn collides(&self, cx: f64, cy: f64, half: f64) -> bool {
        let rx0 = cx - half;
        let ry0 = cy - half;
        let rx1 = cx + half;
        let ry1 = cy + half;

        for w in &self.walls {
            if rx0 < w.x1 && rx1 > w.x0 && ry0 < w.y1 && ry1 > w.y0 {
                return true;
            }
        }
        false
    }

    /// Ray-cast from (cx, cy) in 4 cardinal directions.
    /// Returns the distance to the nearest wall surface in each direction.
    pub fn sense(&self, cx: f64, cy: f64) -> Sensors {
        let mut left = cx;
        let mut right = self.width - cx;
        let mut up = cy;
        let mut down = self.height - cy;

        for w in &self.walls {
            // Left: wall right edge is to our left, and we are within its vertical span
            if w.x1 <= cx && cy > w.y0 && cy < w.y1 {
                left = left.min(cx - w.x1);
            }
            // Right: wall left edge is to our right
            if w.x0 >= cx && cy > w.y0 && cy < w.y1 {
                right = right.min(w.x0 - cx);
            }
            // Up: wall bottom edge is above us
            if w.y1 <= cy && cx > w.x0 && cx < w.x1 {
                up = up.min(cy - w.y1);
            }
            // Down: wall top edge is below us
            if w.y0 >= cy && cx > w.x0 && cx < w.x1 {
                down = down.min(w.y0 - cy);
            }
        }

        Sensors {
            left: Some(left),
            right: Some(right),
            up: Some(up),
            down: Some(down),
        }
    }
}
