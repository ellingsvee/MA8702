use crate::maze::Maze;

pub trait Item {
    /// Position in continuous screen-pixel coordinates (center of the shape).
    fn position(&self) -> (f64, f64);
    fn color(&self) -> u32;
    /// Half-width of the square in screen pixels.
    fn size(&self) -> f64 {
        5.0
    }
}

#[derive(Debug, Clone)]
struct Rect {
    cx: f64,
    cy: f64,
    half: f64,
    color: u32,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    shapes: Vec<Rect>,
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            shapes: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.shapes.clear();
    }

    pub fn add_rect(&mut self, cx: f64, cy: f64, half: f64, color: u32) {
        self.shapes.push(Rect {
            cx,
            cy,
            half,
            color,
        });
    }

    pub fn draw_items(&mut self, items: &[impl Item]) {
        // self.clear();
        for item in items {
            let (x, y) = item.position();
            self.add_rect(x, y, item.size(), item.color());
        }
    }

    pub fn draw_frame(&self, buffer: &mut [u32]) {
        let w = self.width;
        for shape in &self.shapes {
            let h = shape.half;
            let x0 = ((shape.cx - h).floor() as isize).max(0) as usize;
            let y0 = ((shape.cy - h).floor() as isize).max(0) as usize;
            let x1 = ((shape.cx + h).ceil() as usize + 1).min(self.width);
            let y1 = ((shape.cy + h).ceil() as usize + 1).min(self.height);

            for y in y0..y1 {
                let row_start = y * w + x0;
                let row_end = y * w + x1;
                buffer[row_start..row_end].fill(shape.color);
            }
        }
    }

    /// Draw maze walls into the buffer.
    pub fn draw_maze(&self, buffer: &mut [u32], maze: &Maze, color: u32) {
        let w = self.width;
        for wall in &maze.walls {
            let x0 = (wall.x0.floor() as isize).max(0) as usize;
            let y0 = (wall.y0.floor() as isize).max(0) as usize;
            let x1 = (wall.x1.ceil() as usize).min(self.width);
            let y1 = (wall.y1.ceil() as usize).min(self.height);

            for y in y0..y1 {
                let row_start = y * w + x0;
                let row_end = y * w + x1;
                buffer[row_start..row_end].fill(color);
            }
        }
    }

    /// Draw sensor lines from the robot center in active cardinal directions.
    pub fn draw_sensors(
        &self,
        buffer: &mut [u32],
        cx: f64,
        cy: f64,
        left: Option<f64>,
        right: Option<f64>,
        up: Option<f64>,
        down: Option<f64>,
        color: u32,
    ) {
        let w = self.width;
        let icx = cx as usize;
        let icy = cy as usize;

        // Horizontal line segments
        let x0 = left
            .map(|l| (cx - l).ceil() as usize)
            .unwrap_or(icx)
            .max(0)
            .min(self.width);
        let x1 = right
            .map(|r| (cx + r).floor() as usize + 1)
            .unwrap_or(icx + 1)
            .min(self.width);
        if icy < self.height && x0 < x1 {
            buffer[icy * w + x0..icy * w + x1].fill(color);
        }

        // Vertical line segments
        let y0 = up
            .map(|u| (cy - u).ceil() as usize)
            .unwrap_or(icy)
            .max(0)
            .min(self.height);
        let y1 = down
            .map(|d| (cy + d).floor() as usize + 1)
            .unwrap_or(icy + 1)
            .min(self.height);
        if icx < self.width && y0 < y1 {
            for y in y0..y1 {
                buffer[y * w + icx] = color;
            }
        }
    }
}
