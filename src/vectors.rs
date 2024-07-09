use std::{
    fmt,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{delta::DELTA_TOLERANCE, matrix::Matrix};

#[derive(Debug, Clone, Copy)]
pub struct Point;

#[derive(Debug, Clone, Copy)]
pub struct Vector;

pub trait SpaceUnit {}

impl SpaceUnit for Point {}
impl SpaceUnit for Vector {}

#[derive(Debug, Clone, Copy)]
pub struct SpaceElement<T: SpaceUnit> {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    _mark: PhantomData<T>,
}

impl<T: SpaceUnit> SpaceElement<T> {
    pub fn is_sign_positive(&self) -> bool {
        self.x.is_sign_positive() && self.y.is_sign_positive() && self.z.is_sign_positive()
    }
    pub fn is_sign_negative(&self) -> bool {
        self.x.is_sign_negative() || self.y.is_sign_negative() || self.z.is_sign_negative()
    }
}

impl<T: SpaceUnit> Neg for SpaceElement<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::Output {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            _mark: PhantomData,
        }
    }
}

impl<T: SpaceUnit> Mul<f64> for SpaceElement<T> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            _mark: PhantomData,
        }
    }
}

impl<T: SpaceUnit> Mul<SpaceElement<T>> for f64 {
    type Output = SpaceElement<T>;

    fn mul(self, rhs: SpaceElement<T>) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            _mark: PhantomData,
        }
    }
}

impl<T: SpaceUnit> Div<f64> for SpaceElement<T> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::Output {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            _mark: PhantomData,
        }
    }
}

impl<T: SpaceUnit> Div<SpaceElement<T>> for f64 {
    type Output = SpaceElement<T>;

    fn div(self, rhs: SpaceElement<T>) -> Self::Output {
        Self::Output {
            x: self / rhs.x,
            y: self / rhs.y,
            z: self / rhs.z,
            _mark: PhantomData,
        }
    }
}

impl SpaceElement<Point> {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            _mark: PhantomData,
        }
    }

    pub fn as_usized(&self) -> (usize, usize, usize) {
        let x = self.x.round() as usize;
        let y = self.y.round() as usize;
        let z = self.z.round() as usize;

        (x, y, z)
    }

    pub fn to_matrix(&self) -> Matrix<4, 1> {
        Matrix::from([self.x, self.y, self.z, 0.])
    }
}

impl PartialEq for SpaceElement<Point> {
    fn eq(&self, other: &Self) -> bool {
        ((self.x - other.x).abs() < DELTA_TOLERANCE)
            && ((self.y - other.y).abs() < DELTA_TOLERANCE)
            && ((self.z - other.z).abs() < DELTA_TOLERANCE)
    }
}

impl Add<SpaceElement<Vector>> for SpaceElement<Point> {
    type Output = SpaceElement<Point>;

    fn add(self, rhs: SpaceElement<Vector>) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            _mark: PhantomData,
        }
    }
}

impl Add<SpaceElement<Point>> for SpaceElement<Vector> {
    type Output = SpaceElement<Point>;

    fn add(self, rhs: SpaceElement<Point>) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            _mark: PhantomData,
        }
    }
}

impl Sub<SpaceElement<Vector>> for SpaceElement<Point> {
    type Output = SpaceElement<Point>;

    fn sub(self, rhs: SpaceElement<Vector>) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            _mark: PhantomData,
        }
    }
}

impl Sub<SpaceElement<Point>> for SpaceElement<Vector> {
    type Output = SpaceElement<Point>;

    fn sub(self, rhs: SpaceElement<Point>) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            _mark: PhantomData,
        }
    }
}

impl Sub for SpaceElement<Point> {
    type Output = SpaceElement<Vector>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            _mark: PhantomData,
        }
    }
}

impl fmt::Display for SpaceElement<Point> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point({}, {}, {})", self.x, self.y, self.z)
    }
}

impl SpaceElement<Vector> {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            _mark: PhantomData,
        }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.,
            y: 0.,
            z: 0.,
            _mark: PhantomData,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.x == 0. && self.y == 0. && self.z == 0.
    }

    pub fn magnitude(&self) -> f64 {
        (self.x.powf(2.) + self.y.powf(2.) + self.z.powf(2.)).sqrt()
    }

    pub fn normalize(self) -> Self {
        let magnitude = self.magnitude();

        if magnitude > 0. {
            return Self {
                x: self.x / magnitude,
                y: self.y / magnitude,
                z: self.z / magnitude,
                _mark: PhantomData,
            };
        }

        self
    }

    pub fn scalar_product(&self, rhs: &Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross_product(&self, rhs: &Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
            _mark: PhantomData,
        }
    }

    pub fn to_matrix(&self) -> Matrix<4, 1> {
        Matrix::from([self.x, self.y, self.z, 1.])
    }
}

impl PartialEq for SpaceElement<Vector> {
    fn eq(&self, other: &Self) -> bool {
        ((self.x - other.x).abs() < DELTA_TOLERANCE)
            && ((self.y - other.y).abs() < DELTA_TOLERANCE)
            && ((self.z - other.z).abs() < DELTA_TOLERANCE)
    }
}

impl Add for SpaceElement<Vector> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            _mark: PhantomData,
        }
    }
}

impl Sub for SpaceElement<Vector> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            _mark: PhantomData,
        }
    }
}

impl fmt::Display for SpaceElement<Vector> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector({}, {}, {})", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Coordinates {
    pub x: usize, // non octal
    pub y: usize, // non octal
}

impl From<SpaceElement<Point>> for Coordinates {
    fn from(value: SpaceElement<Point>) -> Self {
        let (x, y, _) = value.as_usized();

        Self { x: x + 1, y: y + 1 }
    }
}
impl From<&SpaceElement<Point>> for Coordinates {
    fn from(value: &SpaceElement<Point>) -> Self {
        let (x, y, _) = value.as_usized();

        Self { x: x + 1, y: y + 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    #[test]
    fn point_can_be_created() {
        let point = SpaceElement::<Point>::new(4.3, -4.2, 3.1);
        assert_eq!(point.x, 4.3);
        assert_eq!(point.y, -4.2);
        assert_eq!(point.z, 3.1);
    }

    #[test]
    fn vector_can_be_created() {
        let vector = SpaceElement::<Vector>::new(4.3, -4.2, 3.1);
        assert_eq!(vector.x, 4.3);
        assert_eq!(vector.y, -4.2);
        assert_eq!(vector.z, 3.1);
    }

    #[test]
    fn point_can_be_displayed() {
        let point = SpaceElement::<Point>::new(4.3, -4.2, 3.1);
        assert_eq!(point.to_string(), "Point(4.3, -4.2, 3.1)");
    }

    #[test]
    fn vector_can_be_displayed() {
        let vector = SpaceElement::<Vector>::new(4.3, -4.2, 3.1);
        assert_eq!(vector.to_string(), "Vector(4.3, -4.2, 3.1)");
    }

    #[test]
    fn point_can_be_default() {
        let point = SpaceElement::<Point>::new(0., 0., 0.);

        assert_eq!(point.x, 0.);
        assert_eq!(point.y, 0.);
        assert_eq!(point.z, 0.);
    }

    #[test]
    fn vector_can_be_default() {
        let vector = SpaceElement::<Vector>::zero();

        assert_eq!(vector.x, 0.);
        assert_eq!(vector.y, 0.);
        assert_eq!(vector.z, 0.);
    }

    #[test]
    fn point_can_be_mutated() {
        let mut point = SpaceElement::<Point>::new(4.3, -4.2, 3.1);
        point.x = 4.;
        point.y = 4.;
        point.z = 4.;

        assert_eq!(point.x, 4.);
        assert_eq!(point.y, 4.);
        assert_eq!(point.z, 4.);
    }

    #[test]
    fn vector_can_be_mutated() {
        let mut vector = SpaceElement::<Vector>::new(4.3, -4.2, 3.1);
        vector.x = 4.;
        vector.y = 4.;
        vector.z = 4.;

        assert_eq!(vector.x, 4.);
        assert_eq!(vector.y, 4.);
        assert_eq!(vector.z, 4.);
    }

    #[test]
    fn vector_can_be_added() {
        let vector_1 = SpaceElement::<Vector>::new(3., -2., 5.);
        let vector_2 = SpaceElement::<Vector>::new(-2., 3., 1.);
        let expected = SpaceElement::<Vector>::new(1., 1., 6.);
        assert_eq!(vector_1 + vector_2, expected);
        assert_eq!(vector_2 + vector_1, expected);
    }

    #[test]
    fn point_can_be_substracted_and_results_a_vector() {
        let point_1 = SpaceElement::<Point>::new(3., 2., 1.);
        let point_2 = SpaceElement::<Point>::new(5., 6., 7.);

        let expected = SpaceElement::<Vector>::new(-2., -4., -6.);
        assert_eq!(point_1 - point_2, expected);

        let expected = SpaceElement::<Vector>::new(2., 4., 6.);
        assert_eq!(point_2 - point_1, expected);
    }

    #[test]
    fn vector_can_be_substracted() {
        let vector_1 = SpaceElement::<Vector>::new(3., 2., 1.);
        let vector_2 = SpaceElement::<Vector>::new(5., 6., 7.);

        let expected = SpaceElement::<Vector>::new(-2., -4., -6.);
        assert_eq!(vector_1 - vector_2, expected);

        let expected = SpaceElement::<Vector>::new(2., 4., 6.);
        assert_eq!(vector_2 - vector_1, expected);
    }

    #[test]
    fn vector_can_be_substracted_by_a_point_and_results_a_point() {
        let point = SpaceElement::<Point>::new(3., 2., 1.);
        let vector = SpaceElement::<Vector>::new(5., 6., 7.);

        let expected = SpaceElement::<Point>::new(-2., -4., -6.);
        assert_eq!(point - vector, expected);

        let expected = SpaceElement::<Point>::new(2., 4., 6.);
        assert_eq!(vector - point, expected);
    }

    #[test]
    fn substracting_vector_from_zero_vector() {
        let zero = SpaceElement::<Vector>::zero();
        let vector = SpaceElement::<Vector>::new(5., 6., 7.);
        let expected = SpaceElement::<Vector>::new(-5., -6., -7.);
        assert_eq!(zero - vector, expected);
    }

    #[test]
    fn point_can_be_negated() {
        let point = SpaceElement::<Point>::new(5., 6., 7.);
        let expected = SpaceElement::<Point>::new(-5., -6., -7.);
        assert_eq!(-point, expected);
    }

    #[test]
    fn vector_can_be_negated() {
        let vector = SpaceElement::<Vector>::new(5., 6., 7.);
        let expected = SpaceElement::<Vector>::new(-5., -6., -7.);
        assert_eq!(-vector, expected);
    }

    #[test]
    fn multiply_a_point_by_a_fraction() {
        let point = SpaceElement::<Point>::new(1., -2., 3.);
        let fraction = 0.5;
        let expected = SpaceElement::<Point>::new(0.5, -1., 1.5);
        assert_eq!(point * fraction, expected);
    }

    #[test]
    fn multiply_a_fraction_by_a_point() {
        let point = SpaceElement::<Point>::new(1., -2., 3.);
        let fraction = 0.5;
        let expected = SpaceElement::<Point>::new(0.5, -1., 1.5);
        assert_eq!(fraction * point, expected);
    }

    #[test]
    fn multiply_a_vector_by_a_fraction() {
        let vector = SpaceElement::<Vector>::new(1., -2., 3.);
        let fraction = 0.5;
        let expected = SpaceElement::<Vector>::new(0.5, -1., 1.5);
        assert_eq!(vector * fraction, expected);
    }

    #[test]
    fn multiply_a_fraction_by_a_vector() {
        let vector = SpaceElement::<Vector>::new(1., -2., 3.);
        let fraction = 0.5;
        let expected = SpaceElement::<Vector>::new(0.5, -1., 1.5);
        assert_eq!(fraction * vector, expected);
    }

    #[test]
    fn divide_a_point_by_a_scalar() {
        let point = SpaceElement::<Point>::new(1., -2., 3.);
        let scalar = 0.5;
        let expected = SpaceElement::<Point>::new(2., -4., 6.);
        assert_eq!(point / scalar, expected);
    }

    #[test]
    fn divide_a_scalar_by_a_point() {
        let point = SpaceElement::<Point>::new(1., -2., 3.);
        let scalar = 0.5;
        let expected = SpaceElement::<Point>::new(0.5, -0.25, 0.5 / 3.);
        assert_eq!(scalar / point, expected);
    }

    #[test]
    fn divide_a_vector_by_a_scalar() {
        let vector = SpaceElement::<Vector>::new(1., -2., 3.);
        let scalar = 0.5;
        let expected = SpaceElement::<Vector>::new(2., -4., 6.);
        assert_eq!(vector / scalar, expected);
    }

    #[test]
    fn divide_a_scalar_by_a_vector() {
        let vector = SpaceElement::<Vector>::new(1., -2., 3.);
        let scalar = 0.5;
        let expected = SpaceElement::<Vector>::new(0.5, -0.25, 0.5 / 3.);
        assert_eq!(scalar / vector, expected);
    }

    #[test]
    fn compute_the_magnitude_of_vectors() {
        let vector = SpaceElement::<Vector>::new(1., 0., 0.);
        assert_eq!(1., vector.magnitude());

        let vector = SpaceElement::<Vector>::new(0., 1., 0.);
        assert_eq!(1., vector.magnitude());

        let vector = SpaceElement::<Vector>::new(0., 0., 1.);
        assert_eq!(1., vector.magnitude());

        let vector = SpaceElement::<Vector>::new(1., 2., 3.);
        assert_delta!(3.74165, vector.magnitude(), DELTA_TOLERANCE);

        let vector = SpaceElement::<Vector>::new(-1., -2., -3.);
        assert_delta!(3.74165, vector.magnitude(), DELTA_TOLERANCE);
    }

    #[test]
    fn normalize_vectors() {
        let vector = SpaceElement::<Vector>::new(4., 0., 0.).normalize();
        assert_eq!(vector, SpaceElement::<Vector>::new(1., 0., 0.));

        let vector = SpaceElement::<Vector>::new(1., 2., 3.).normalize();
        assert_eq!(
            vector,
            SpaceElement::<Vector>::new(0.26726, 0.53452, 0.80178)
        );
    }

    #[test]
    fn magnitude_of_normalize_vector_is_1() {
        let vector = SpaceElement::<Vector>::new(4., 0., 0.).normalize();
        assert_eq!(1., vector.magnitude());
    }

    #[test]
    fn scalar_product_of_two_vectors() {
        let vector_1 = SpaceElement::<Vector>::new(1., 2., 3.);
        let vector_2 = SpaceElement::<Vector>::new(2., 3., 4.);
        assert_eq!(20., vector_1.scalar_product(&vector_2));
    }

    #[test]
    fn cross_product_of_two_vectors() {
        let vector_1 = SpaceElement::<Vector>::new(1., 2., 3.);
        let vector_2 = SpaceElement::<Vector>::new(2., 3., 4.);

        let expect = SpaceElement::<Vector>::new(-1., 2., -1.);
        assert_eq!(expect, vector_1.cross_product(&vector_2));

        let expect = SpaceElement::<Vector>::new(1., -2., 1.);
        assert_eq!(expect, vector_2.cross_product(&vector_1));
    }
}
