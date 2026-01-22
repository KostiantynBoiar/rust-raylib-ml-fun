pub enum Loss{
    SumSquaredError
}

impl Loss{
    pub fn calculate(&self, predicted: f64, actual: f64) -> f64 {
        match self {
            Loss::SumSquaredError => sum_squared_error(predicted, actual),
        }
    }
}

pub fn sum_squared_error(predicted: f64, actual: f64) -> f64 {
    return ((predicted - actual).powi(2)) / 2.;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_squared_error() {
        assert_eq!(sum_squared_error(1.0, 2.0), 0.5);
    }
}