use derive_more::Display;

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
pub struct Matrix<const ROWS: usize, const COLS: usize>
where
    [f64; ROWS * COLS]:,
{
    pub rows: usize,
    pub cols: usize,
    pub data: [f64; ROWS * COLS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]:,
{
    pub fn new() -> Self {
        Self {
            rows: ROWS,
            cols: COLS,
            data: [0.; ROWS * COLS],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        let position = self.get_position(row, col)?;
        self.data.get(position)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        let position = self.get_position(row, col)?;
        self.data.get_mut(position)
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

    fn get_position(&self, row: usize, col: usize) -> Option<usize> {
        if (row > self.rows || row == 0) || (col > self.cols || col == 0) {
            return None;
        }

        let position = row * self.cols - (self.cols - col) - 1;

        if position < self.data.len() {
            return Some(position);
        }

        None
    }

    pub fn get_row(&self, row: usize) -> Result<[f64; COLS], Error> {
        if row > ROWS || row == 0 {
            return Err(Error::OverflowRow(row));
        }

        let mut data: [f64; COLS] = [0.; COLS];

        let origin_position = (row * self.cols) - self.cols;
        let target_position = row * self.cols;

        if origin_position >= self.data.len() {
            return Err(Error::OverflowIndex(origin_position));
        }
        if target_position > self.data.len() {
            return Err(Error::OverflowIndex(target_position));
        }
        if target_position - origin_position != COLS {
            return Err(Error::OverflowIndex(target_position));
        }

        for (index, element) in self.data.as_slice()[origin_position..target_position]
            .iter()
            .enumerate()
        {
            data[index] = *element;
        }

        Ok(data)
    }

    pub fn set_row(&mut self, row: usize, data: [f64; COLS]) -> Result<(), Error> {
        if row > ROWS || row == 0 {
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
        if target_position - origin_position != COLS {
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

    pub fn get_col(&self, col: usize) -> Result<[f64; ROWS], Error> {
        if col > COLS || col == 0 {
            return Err(Error::OverflowCol(col));
        }

        let mut positions: [usize; ROWS] = [0; ROWS];

        for row in 1..=ROWS {
            positions[row - 1] = row * COLS - (COLS - col) - 1;
        }

        let mut data: [f64; ROWS] = [0.; ROWS];

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

    pub fn set_col(&mut self, col: usize, data: [f64; ROWS]) -> Result<(), Error> {
        if col > COLS || col == 0 {
            return Err(Error::OverflowCol(col));
        }

        let mut positions: [usize; ROWS] = [0; ROWS];

        for row in 1..=ROWS {
            positions[row - 1] = row * COLS - (COLS - col) - 1;
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

impl<const ROWS: usize, const COLS: usize> From<[f64; ROWS * COLS]> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]:,
{
    fn from(value: [f64; ROWS * COLS]) -> Self {
        Self {
            rows: ROWS,
            cols: COLS,
            data: value,
        }
    }
}

impl<const ROWS: usize, const COLS: usize> PartialEq for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]:,
{
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for position in 0..ROWS * COLS {
            if (self.data[position] - other.data[position]).abs() >= DELTA_TOLERANCE {
                return false;
            }
        }

        true
    }
}

impl<const ROWS: usize, const COLS: usize> std::fmt::Display for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]:,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_can_be_created() {
        let matrix = Matrix::<10, 10>::new();

        assert_eq!(matrix.get(1, 1), Some(&0f64));
        assert_eq!(matrix.get(2, 2), Some(&0f64));
        assert_eq!(matrix.get(3, 3), Some(&0f64));
        assert_eq!(matrix.get(4, 4), Some(&0f64));
        assert_eq!(matrix.get(5, 5), Some(&0f64));
        assert_eq!(matrix.get(6, 6), Some(&0f64));
        assert_eq!(matrix.get(7, 7), Some(&0f64));
        assert_eq!(matrix.get(8, 8), Some(&0f64));
        assert_eq!(matrix.get(9, 9), Some(&0f64));
        assert_eq!(matrix.get(10, 10), Some(&0f64));
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
            0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.,
        ]);

        assert_eq!(matrix.get_row(1), Ok([1., 1., 1.]));
        assert_eq!(matrix.get_row(2), Ok([0., 0., 0.]));
        assert_eq!(matrix.get_row(3), Ok([0., 0., 0.]));
        assert_eq!(matrix.get_row(4), Ok([0., 0., 0.]));
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
            0., 0., 1., 0.,
            0., 0., 1., 0.,
            0., 0., 1., 0.,
        ]);

        assert_eq!(matrix.get_col(1), Ok([0., 0., 0.,]));
        assert_eq!(matrix.get_col(2), Ok([0., 0., 0.]));
        assert_eq!(matrix.get_col(3), Ok([1., 1., 1.]));
        assert_eq!(matrix.get_col(4), Ok([0., 0., 0.]));
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
}
