use derive_more::Display;

use crate::delta::DELTA_TOLERANCE;

#[derive(Debug, Display, Clone, Copy, PartialEq)]
pub enum Error {
    #[display(fmt = "Write overflow at index {}", _0)]
    WriteOverflowIndex(usize),
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

    pub fn get(&self, row: usize, col: usize) -> Option<(usize, f64)> {
        let position = self.get_position(row, col)?;
        Some((position, self.data[position]))
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<(usize, &mut f64)> {
        let position = self.get_position(row, col)?;
        Some((position, &mut self.data[position]))
    }

    pub fn set_row(&mut self, row: usize, data: [f64; COLS]) -> Result<(), Error> {
        let origin_position = (row * self.cols) - self.cols;
        let target_position = row * self.cols;

        if origin_position >= self.data.len() {
            return Err(Error::WriteOverflowIndex(origin_position));
        }
        if target_position >= self.data.len() {
            return Err(Error::WriteOverflowIndex(target_position));
        }
        if target_position - origin_position != 4 {
            return Err(Error::WriteOverflowIndex(target_position));
        }

        let mut data = data.into_iter();

        for position in origin_position..target_position {
            if let Some(value) = data.next() {
                self.data[position] = value;
            }
        }

        Ok(())
    }

    pub fn identity(mut self) -> Self {
        if self.rows != self.cols {
            return self;
        }

        let mut position: usize = 0;

        while position <= (self.rows * self.cols) - 1 {
            self.data[position] = 1.;
            position += self.cols + 1;
        }

        self
    }

    fn get_position(&self, row: usize, col: usize) -> Option<usize> {
        if row > self.rows || row == 0 {
            return None;
        }
        if col > self.cols || col == 0 {
            return None;
        }

        let position = row * self.cols - (self.cols - col) - 1;

        if position > self.data.len() - 1 {
            return None;
        }

        Some(position)
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
    use super::Matrix;

    #[test]
    fn matrix_can_be_created() {
        let matrix = Matrix::<10, 10>::new();

        assert_eq!(matrix.get(1, 1), Some((0, 0f64)));
        assert_eq!(matrix.get(2, 2), Some((11, 0f64)));
        assert_eq!(matrix.get(3, 3), Some((22, 0f64)));
        assert_eq!(matrix.get(4, 4), Some((33, 0f64)));
        assert_eq!(matrix.get(5, 5), Some((44, 0f64)));
        assert_eq!(matrix.get(6, 6), Some((55, 0f64)));
        assert_eq!(matrix.get(7, 7), Some((66, 0f64)));
        assert_eq!(matrix.get(8, 8), Some((77, 0f64)));
        assert_eq!(matrix.get(9, 9), Some((88, 0f64)));
        assert_eq!(matrix.get(10, 10), Some((99, 0f64)));
    }

    #[test]
    fn matrix_can_be_identity() {
        let matrix = Matrix::<10, 10>::new().identity();

        assert_eq!(matrix.get(1, 1), Some((0, 1f64)));
        assert_eq!(matrix.get(2, 2), Some((11, 1f64)));
        assert_eq!(matrix.get(3, 3), Some((22, 1f64)));
        assert_eq!(matrix.get(4, 4), Some((33, 1f64)));
        assert_eq!(matrix.get(5, 5), Some((44, 1f64)));
        assert_eq!(matrix.get(6, 6), Some((55, 1f64)));
        assert_eq!(matrix.get(7, 7), Some((66, 1f64)));
        assert_eq!(matrix.get(8, 8), Some((77, 1f64)));
        assert_eq!(matrix.get(9, 9), Some((88, 1f64)));
        assert_eq!(matrix.get(10, 10), Some((99, 1f64)));
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
            0., 0., 0., 0.,
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
    fn matrix_can_set_row() {
        #[rustfmt::skip]
        let mut matrix = Matrix::<4, 4>::from([
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        ]);

        let row: [f64; 4] = [1., 1., 1., 1.];
        matrix.set_row(2, row).unwrap();

        #[rustfmt::skip]
        assert_eq!(matrix, Matrix::<4, 4>::from([
            0., 0., 0., 0.,
            1., 1., 1., 1.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        ]));
    }
}
