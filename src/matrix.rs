use derive_more::Display;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::ops::{Add, Mul, Neg};

use crate::delta::DELTA_TOLERANCE;

#[derive(Debug, Display, Clone, Copy, PartialEq)]
pub enum Error {
    #[display(fmt = "Matrix overflow at index {}", _0)]
    OverflowIndex(usize),

    #[display(fmt = "Matrix overflow at row {}", _0)]
    OverflowRow(usize),

    #[display(fmt = "Matrix overflow at col {}", _0)]
    OverflowCol(usize),
}

#[derive(Debug, Clone)]
pub struct Matrix<const M: usize, const N: usize>
where
    [f64; M * N]:,
{
    pub rows: usize,
    pub cols: usize,
    pub data: [f64; M * N],
}

impl<const M: usize, const N: usize> Matrix<M, N>
where
    [f64; M * N]:,
{
    pub fn new() -> Self {
        Self {
            rows: M,
            cols: N,
            data: [0.; M * N],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.data.par_iter().all(|element| *element == 0.)
    }

    pub fn is_square(&self) -> bool {
        self.cols == self.rows
    }

    pub fn is_simetric(&self) -> bool {
        *self == self.clone().transpose()
    }

    pub fn is_antisimetric(&self) -> bool {
        *self == -self.clone().transpose()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        let position = self.get_position(row, col)?;
        self.data.get(position)
    }

    pub fn identity(mut self) -> Self {
        if self.rows != self.cols {
            return self;
        }

        let mut position: usize = 0;
        while position < self.data.len() {
            self.data[position] = 1.;
            position += self.cols + 1;
        }

        self
    }

    pub fn get_position(&self, row: usize, col: usize) -> Option<usize> {
        if (row > self.rows || row == 0) || (col > self.cols || col == 0) {
            return None;
        }

        let position = row * self.cols - (self.cols - col) - 1;

        if position < self.data.len() {
            return Some(position);
        }

        None
    }

    pub fn get_row_col(&self, position: usize) -> Option<(usize, usize)> {
        if position >= self.data.len() {
            return None;
        }

        let mut col = 1;
        let mut row = 1;

        for n in 0..self.data.len() {
            if col > self.cols {
                row += 1;
                col = 1;
            }

            if n == position {
                return Some((row, col));
            }

            col += 1;
        }

        None
    }

    pub fn transpose<const P: usize, const Q: usize>(self) -> Matrix<P, Q>
    where
        [f64; P * Q]:,
    {
        let mut matrix = Matrix::<P, Q>::new();

        for position in 0..self.data.len() {
            let Some((row, col)) = self.get_row_col(position) else {
                continue;
            };
            let Some(target_position) = matrix.get_position(col, row) else {
                continue;
            };

            matrix.data[target_position] = self.data[position];
        }

        matrix
    }

    pub fn get_row(&self, row: usize) -> Result<[f64; N], Error> {
        if row > self.rows || row == 0 {
            return Err(Error::OverflowRow(row));
        }

        let mut row_arr: [f64; N] = [0.; N];

        let origin_position = (row * self.cols) - self.cols;
        let target_position = row * self.cols;

        if origin_position >= self.data.len() {
            return Err(Error::OverflowIndex(origin_position));
        }
        if target_position > self.data.len() {
            return Err(Error::OverflowIndex(target_position));
        }
        if target_position - origin_position != N {
            return Err(Error::OverflowIndex(target_position));
        }

        let mut insert_index = 0;
        for n in origin_position..target_position {
            if insert_index < N {
                row_arr[insert_index] = self.data[n];
                insert_index += 1;
            }
        }

        Ok(row_arr)
    }

    pub fn set_row(&mut self, row: usize, data: [f64; N]) -> Result<(), Error> {
        if row > self.rows || row == 0 {
            return Err(Error::OverflowRow(row));
        }

        let origin_position = (row * self.cols) - self.cols;
        let target_position = row * self.cols;

        if origin_position >= self.data.len() {
            return Err(Error::OverflowIndex(origin_position));
        }
        if target_position > self.data.len() {
            return Err(Error::OverflowIndex(target_position));
        }
        if target_position - origin_position != N {
            return Err(Error::OverflowIndex(target_position));
        }

        let mut data = data.into_iter();

        for position in origin_position..target_position {
            if let Some(value) = data.next() {
                self.data[position] = value;
            }
        }

        Ok(())
    }

    pub fn get_col(&self, col: usize) -> Result<[f64; M], Error> {
        if col > self.cols || col == 0 {
            return Err(Error::OverflowCol(col));
        }

        let mut positions: [usize; M] = [0; M];

        for row in 1..=M {
            positions[row - 1] = row * N - (N - col) - 1;
        }

        let mut data: [f64; M] = [0.; M];

        for n in 0..positions.len() {
            let position = positions[n];

            if position >= self.data.len() {
                return Err(Error::OverflowIndex(position));
            }

            if n >= data.len() {
                return Err(Error::OverflowIndex(n));
            }

            data[n] = self.data[positions[n]];
        }

        Ok(data)
    }

    pub fn set_col(&mut self, col: usize, data: [f64; M]) -> Result<(), Error> {
        if col > self.cols || col == 0 {
            return Err(Error::OverflowCol(col));
        }

        let mut positions: [usize; M] = [0; M];

        for row in 1..=M {
            positions[row - 1] = row * N - (N - col) - 1;
        }

        let mut data = data.into_iter();

        for position in positions {
            if position >= self.data.len() {
                return Err(Error::OverflowIndex(position));
            }

            if let Some(value) = data.next() {
                self.data[position] = value;
            }
        }

        Ok(())
    }
}

impl<const M: usize, const N: usize> From<[f64; M * N]> for Matrix<M, N>
where
    [f64; M * N]:,
{
    fn from(value: [f64; M * N]) -> Self {
        Self {
            rows: M,
            cols: N,
            data: value,
        }
    }
}

impl<const M: usize, const N: usize> PartialEq for Matrix<M, N>
where
    [f64; M * N]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.data
            .par_iter()
            .enumerate()
            .all(|(idx, el)| (el - other.data[idx]).abs() < DELTA_TOLERANCE)
    }
}

impl<const M: usize, const N: usize> Neg for Matrix<M, N>
where
    [f64; M * N]:,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.data.par_iter_mut().for_each(|p| *p = -*p);
        self
    }
}

impl<const M: usize, const N: usize> std::fmt::Display for Matrix<M, N>
where
    [f64; M * N]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut str = String::new();
        for chunk in self.data.chunks(self.cols) {
            let mut values = vec![String::default(); chunk.len()];
            for inner in chunk {
                values.push(format!("{inner}"));
            }
            str += values.join("  ").as_str();
            str.push('\n')
        }
        write!(f, "{str}")
    }
}

impl<const M: usize, const N: usize> Add for Matrix<M, N>
where
    [f64; M * N]:,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.data
            .par_iter_mut()
            .enumerate()
            .for_each(|(position, element)| {
                *element = *element + rhs.data[position];
            });

        self
    }
}

impl<const M: usize, const N: usize> Mul<f64> for Matrix<M, N>
where
    [f64; M * N]:,
{
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.data.par_iter_mut().for_each(|element| {
            *element = *element + rhs;
        });

        self
    }
}

impl<const M: usize, const N: usize> Mul<Matrix<M, N>> for f64
where
    [f64; M * N]:,
{
    type Output = Matrix<M, N>;

    fn mul(self, mut rhs: Matrix<M, N>) -> Self::Output {
        rhs.data.par_iter_mut().for_each(|element| {
            *element = *element + self;
        });

        rhs
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N>
where
    [f64; M * N]:,
    [f64; N * P]:,
    [f64; M * P]:,
{
    type Output = Matrix<M, P>;

    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut next = Matrix::<M, P>::new();

        let mut position = 0;

        for n in 1..=self.rows {
            let Ok(row) = self.get_row(n) else {
                return next;
            };

            for m in 1..=rhs.cols {
                let Ok(col) = rhs.get_col(m) else {
                    return next;
                };

                next.data[position] = std::iter::zip(row, col).map(|(a, b)| a * b).sum();

                position += 1;
            }
        }

        next
    }
}

#[cfg(test)]
mod tests {
    use crate::vectors::{SpaceElement, Vector};

    use super::*;

    #[test]
    fn matrix_can_be_created() {
        let matrix = Matrix::<10, 10>::new();

        #[rustfmt::skip]
        assert_eq!(matrix.data, [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]);
    }

    #[test]
    fn matrix_can_be_compared() {
        #[rustfmt::skip]
        let matrix_1 = Matrix::<4, 4>::from([
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        ]);

        #[rustfmt::skip]
        let matrix_2 = Matrix::<4, 4>::from([
            0., 0., 0., 0.0000078,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        ]);

        assert_eq!(matrix_1, matrix_2);

        #[rustfmt::skip]
        let matrix_1 = Matrix::<4, 4>::from([
            1., 0., 0., 0.,
            3., 0., 0.4, 0.8,
            5., 0., 0., 0.,
            7., 0., 0., 0.,
        ]);

        #[rustfmt::skip]
        let matrix_2 = Matrix::<4, 4>::from([
            2., 0., 0., 0.,
            4., 0., 0.6, 0.9,
            6., 0., 0., 0.,
            8., 0., 0., 0.,
        ]);

        assert_ne!(matrix_1, matrix_2);
    }

    #[test]
    fn matrix_can_be_identity() {
        let matrix = Matrix::<10, 10>::new().identity();

        #[rustfmt::skip]
        assert_eq!(matrix.data, [
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        ]);
    }

    #[test]
    fn matrix_can_get_row() {
        #[rustfmt::skip]
        let matrix = Matrix::<4, 3>::from([
            1., 1., 1.,
            0., 1., 0.,
            0., 0., 1.,
            1., 0., 0.,
        ]);

        assert_eq!(matrix.get_row(1), Ok([1., 1., 1.]));
        assert_eq!(matrix.get_row(2), Ok([0., 1., 0.]));
        assert_eq!(matrix.get_row(3), Ok([0., 0., 1.]));
        assert_eq!(matrix.get_row(4), Ok([1., 0., 0.]));
    }

    #[test]
    fn matrix_cant_get_row_if_overflow() {
        #[rustfmt::skip]
        let matrix = Matrix::<4, 3>::from([
            1., 1., 1.,
            0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.,
        ]);

        assert_eq!(matrix.get_row(0), Err(Error::OverflowRow(0)));
        assert_eq!(matrix.get_row(5), Err(Error::OverflowRow(5)));
    }

    #[test]
    fn matrix_can_set_row() {
        #[rustfmt::skip]
        let mut matrix = Matrix::<2, 4>::from([
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        ]);

        matrix.set_row(1, [2., 0., 0., 0.]).unwrap();
        matrix.set_row(2, [1., 1., 1., 1.]).unwrap();

        #[rustfmt::skip]
        assert_eq!(matrix, Matrix::<2, 4>::from([
            2., 0., 0., 0.,
            1., 1., 1., 1.,
        ]));
    }

    #[test]
    fn matrix_cant_set_row_if_overflow() {
        #[rustfmt::skip]
        let mut matrix = Matrix::<4, 3>::from([
            1., 1., 1.,
            0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.,
        ]);

        let row: [f64; 3] = [1., 1., 1.];

        assert_eq!(matrix.set_row(0, row), Err(Error::OverflowRow(0)));
        assert_eq!(matrix.set_row(5, row), Err(Error::OverflowRow(5)));
    }

    #[test]
    fn matrix_can_get_col() {
        #[rustfmt::skip]
        let matrix = Matrix::<3, 4>::from([
            1., 0., 1., 0.,
            0., 1., 1., 0.,
            0., 0., 1., 1.,
        ]);

        assert_eq!(matrix.get_col(1), Ok([1., 0., 0.,]));
        assert_eq!(matrix.get_col(2), Ok([0., 1., 0.]));
        assert_eq!(matrix.get_col(3), Ok([1., 1., 1.]));
        assert_eq!(matrix.get_col(4), Ok([0., 0., 1.]));
    }

    #[test]
    fn matrix_cant_get_col_if_overflow() {
        #[rustfmt::skip]
        let matrix = Matrix::<3, 4>::from([
            0., 0., 1., 0.,
            0., 0., 1., 0.,
            0., 0., 1., 0.,
        ]);

        assert_eq!(matrix.get_col(0), Err(Error::OverflowCol(0)));
        assert_eq!(matrix.get_col(5), Err(Error::OverflowCol(5)));
    }

    #[test]
    fn matrix_can_set_col() {
        #[rustfmt::skip]
        let mut matrix = Matrix::<4, 2>::from([
            0., 0.,
            0., 0.,
            0., 0.,
            0., 0.,
        ]);

        matrix.set_col(1, [2., 0., 0., 0.]).unwrap();
        matrix.set_col(2, [1., 1., 1., 1.]).unwrap();

        #[rustfmt::skip]
        assert_eq!(matrix, Matrix::<4, 2>::from([
            2., 1.,
            0., 1.,
            0., 1.,
            0., 1.,
        ]));
    }

    #[test]
    fn matrix_cant_set_col_if_overflow() {
        #[rustfmt::skip]
        let mut matrix = Matrix::<4, 2>::from([
            0., 0.,
            0., 0.,
            0., 0.,
            0., 0.,
        ]);

        #[rustfmt::skip]
        let col: [f64; 4] = [
            1.,
            1.,
            1.,
            1.
        ];

        assert_eq!(matrix.set_col(0, col), Err(Error::OverflowCol(0)));
        assert_eq!(matrix.set_col(3, col), Err(Error::OverflowCol(3)));
    }

    #[test]
    fn matrix_can_extract_rowcol_from_position() {
        #[rustfmt::skip]
        let matrix = Matrix::<3, 3>::from([
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
        ]);

        assert_eq!(Some((1, 1)), matrix.get_row_col(0));
        assert_eq!(Some((1, 2)), matrix.get_row_col(1));
        assert_eq!(Some((1, 3)), matrix.get_row_col(2));
        assert_eq!(Some((2, 1)), matrix.get_row_col(3));
        assert_eq!(Some((2, 2)), matrix.get_row_col(4));
        assert_eq!(Some((2, 3)), matrix.get_row_col(5));
        assert_eq!(Some((3, 1)), matrix.get_row_col(6));
        assert_eq!(Some((3, 2)), matrix.get_row_col(7));
        assert_eq!(Some((3, 3)), matrix.get_row_col(8));

        #[rustfmt::skip]
        let matrix = Matrix::<3, 1>::from([
            1.,
            4.,
            7.,
        ]);

        assert_eq!(Some((1, 1)), matrix.get_row_col(0));
        assert_eq!(Some((2, 1)), matrix.get_row_col(1));
        assert_eq!(Some((3, 1)), matrix.get_row_col(2));

        #[rustfmt::skip]
        let matrix = Matrix::<1, 3>::from([
            1., 4., 7.,
        ]);

        assert_eq!(Some((1, 1)), matrix.get_row_col(0));
        assert_eq!(Some((1, 2)), matrix.get_row_col(1));
        assert_eq!(Some((1, 3)), matrix.get_row_col(2));
    }

    #[test]
    fn matrix_can_be_transposed() {
        #[rustfmt::skip]
        let matrix = Matrix::<3, 2>::from([
            1., 2.,
            3., 4.,
            5., 6.,
        ]);

        #[rustfmt::skip]
        let transposed = Matrix::<2, 3>::from([
            1., 3., 5.,
            2., 4., 6.
        ]);

        assert_eq!(matrix.transpose(), transposed);

        #[rustfmt::skip]
        let matrix = Matrix::<2, 2>::from([
            1., 2.,
            3., 4.,
        ]);

        #[rustfmt::skip]
        let transposed = Matrix::<2, 2>::from([
            1., 3.,
            2., 4.,
        ]);

        assert_eq!(matrix.transpose(), transposed);

        #[rustfmt::skip]
        let matrix = Matrix::<1, 2>::from([
            1., 2.,
        ]);

        #[rustfmt::skip]
        let transposed = Matrix::<2, 1>::from([
            1.,
            2.,
        ]);

        assert_eq!(matrix.transpose(), transposed);
    }

    #[test]
    fn matrix_can_be_multiplied() {
        #[rustfmt::skip]
        let a = Matrix::<2, 3>::from([
             0.,  4., -2.,
            -4., -3.,  0.
        ]);

        #[rustfmt::skip]
        let b = Matrix::<3, 2>::from([
            0.,  1.,
            1., -1.,
            2.,  3.,
        ]);

        #[rustfmt::skip]
        let c = Matrix::<2, 2>::from([
             0.,  -10.,
            -3.,  -1.,
        ]);

        assert_eq!(c, a * b);
    }

    #[test]
    fn matrix_is_square() {
        #[rustfmt::skip]
        let a = Matrix::<2, 2>::from([
            1., 2.,
            4., 1.,
        ]);

        assert!(a.is_square());
    }

    #[test]
    fn matrix_is_zero() {
        #[rustfmt::skip]
        let a = Matrix::<2, 2>::from([
            0., 0.,
            0., 0.,
        ]);

        assert!(a.is_zero());
    }

    #[test]
    fn matrix_can_be_multiplied_by_a_vector() {
        #[rustfmt::skip]
        let matrix = Matrix::<2, 4>::from([
            1., 2., 3., -3.,
            4., 1., 0.,  0.
        ]);

        #[rustfmt::skip]
        let vector = SpaceElement::<Vector>::new(1., 2., 3.).to_matrix();

        #[rustfmt::skip]
        let result = Matrix::<2, 1>::from([
            11.,
            6.,
        ]);

        assert_eq!(result, matrix * vector);
    }
}
