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
}
