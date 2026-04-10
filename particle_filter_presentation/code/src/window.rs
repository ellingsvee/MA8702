pub trait Item {
    /// Position in continuous screen-pixel coordinates (center of the shape).
    fn position(&self) -> (f64, f64);
    fn color(&self) -> u32;
    /// Half-width of the square in screen pixels.
    fn size(&self) -> f64 {
        5.0
    }
}
