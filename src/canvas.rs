use derive_more::Display;
use std::{
    fmt,
    ops::{Add, Mul, Sub},
};

use crate::{delta::DELTA_TOLERANCE, vectorial_space::Coordinates};

#[derive(Debug, Display, Clone, Copy, PartialEq)]
pub enum Error {
    #[display(fmt = "Write overflow at index {}", _0)]
    WriteOverflowIndex(usize),

    #[display(fmt = "Write overflow at position ({},{})", _0, _1)]
    WriteOverflowCoordinates(usize, usize),

    #[display(fmt = "Write overflow at point ({},{})", _0, _1)]
    WriteOverflowPoint(f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color({}, {}, {})", self.r, self.g, self.b)
    }
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn r_as_byte(&self) -> u8 {
        (self.r.min(1.).max(0.) * 255.999).round() as u8
    }

    pub fn g_as_byte(&self) -> u8 {
        (self.g.min(1.).max(0.) * 255.999).round() as u8
    }

    pub fn b_as_byte(&self) -> u8 {
        (self.b.min(1.).max(0.) * 255.999).round() as u8
    }

    pub fn white() -> Self {
        Self {
            r: 1.,
            g: 1.,
            b: 1.,
        }
    }

    pub fn black() -> Self {
        Self {
            r: 0.,
            g: 0.,
            b: 0.,
        }
    }

    pub fn red() -> Self {
        Self {
            r: 1.,
            g: 0.,
            b: 0.,
        }
    }

    pub fn green() -> Self {
        Self {
            r: 0.,
            g: 1.,
            b: 0.,
        }
    }

    pub fn blue() -> Self {
        Self {
            r: 0.,
            g: 0.,
            b: 1.,
        }
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        ((self.r - other.r) < DELTA_TOLERANCE)
            && ((self.g - other.g) < DELTA_TOLERANCE)
            && ((self.b - other.b) < DELTA_TOLERANCE)
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            r: self.r - rhs.r,
            g: self.g - rhs.g,
            b: self.b - rhs.b,
        }
    }
}

impl Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl Mul<f64> for Color {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}
impl Mul<Color> for f64 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Self::Output {
            r: self * rhs.r,
            g: self * rhs.g,
            b: self * rhs.b,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Canvas<const WIDTH: usize, const HEIGHT: usize>
where
    [Color; WIDTH * HEIGHT]:,
{
    pub width: usize,
    pub height: usize,
    pub pixels: [Color; WIDTH * HEIGHT],
}

impl<const WIDTH: usize, const HEIGHT: usize> Canvas<WIDTH, HEIGHT>
where
    [Color; WIDTH * HEIGHT]:,
{
    pub fn new() -> Self {
        Self {
            width: WIDTH,
            height: HEIGHT,
            pixels: [Color::black(); WIDTH * HEIGHT],
        }
    }

    pub fn clean(&mut self) {
        for pixel in self.pixels.iter_mut() {
            *pixel = Color::black();
        }
    }

    pub fn get_position(&self, Coordinates { x, y }: &Coordinates) -> Result<usize, Error> {
        if *y > self.height || *x > self.width {
            return Err(Error::WriteOverflowCoordinates(*x, *y));
        }

        Ok(((self.height - y) * self.width + x) - 1)
    }

    pub fn write_position(&mut self, position: &usize, color: &Color) -> Result<(), Error> {
        if *position >= self.pixels.len() {
            return Err(Error::WriteOverflowIndex(*position));
        }

        self.pixels[*position] = *color;

        Ok(())
    }

    pub fn write_coordinates(
        &mut self,
        coordinates: &Coordinates,
        color: &Color,
    ) -> Result<(), Error> {
        self.write_position(&self.get_position(coordinates)?, color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_are_red_green_blue_tuples() {
        let color = Color::new(-0.5, 0.4, 1.7);
        assert_eq!(color.r, -0.5);
        assert_eq!(color.g, 0.4);
        assert_eq!(color.b, 1.7);
    }

    #[test]
    fn colors_can_be_added() {
        let color_1 = Color::new(0.9, 0.6, 0.75);
        let color_2 = Color::new(0.7, 0.1, 0.25);
        assert_eq!(color_1 + color_2, Color::new(1.6, 0.7, 1.));
        assert_eq!(color_2 + color_1, Color::new(1.6, 0.7, 1.));
    }

    #[test]
    fn colors_can_be_substracted() {
        let color_1 = Color::new(0.9, 0.6, 0.75);
        let color_2 = Color::new(0.7, 0.1, 0.25);
        assert_eq!(color_1 - color_2, Color::new(0.2, 0.5, 0.5));
        assert_eq!(color_2 - color_1, Color::new(0.2, 0.5, 0.5));
    }

    #[test]
    fn colors_can_be_multiplied_by_scalar() {
        let color = Color::new(0.2, 0.3, 0.4);
        let scalar = 2.;
        assert_eq!(color * scalar, Color::new(0.4, 0.6, 0.8));
        assert_eq!(scalar * color, Color::new(0.4, 0.6, 0.8));
    }

    #[test]
    fn colors_can_be_multiplied() {
        let color_1 = Color::new(1., 0.2, 0.4);
        let color_2 = Color::new(0.9, 1., 0.1);
        assert_eq!(color_1 * color_2, Color::new(0.9, 0.2, 0.04));
        assert_eq!(color_2 * color_1, Color::new(0.9, 0.2, 0.04));
    }

    #[test]
    fn canvas_can_be_created() {
        let canvas = Canvas::<10, 20>::new();
        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);
        assert!(canvas.pixels.iter().all(|pixel| *pixel == Color::black()));
    }

    #[test]
    fn canvas_can_be_writted() {
        let mut canvas = Canvas::<10, 20>::new();
        let coordinates = Coordinates { x: 2, y: 3 };
        let color = Color::red();
        canvas.write_coordinates(&coordinates, &color).unwrap();
        assert_eq!(
            Some(&color),
            canvas
                .pixels
                .get(canvas.get_position(&coordinates).unwrap())
        );
    }

    #[test]
    fn canvas_cant_be_writted_with_overflow_position() {
        let mut canvas = Canvas::<10, 20>::new();
        let color = Color::red();
        assert_eq!(
            Err(Error::WriteOverflowCoordinates(200, 300)),
            canvas.write_coordinates(&Coordinates { x: 200, y: 300 }, &color)
        );
    }
}
