#[derive(Clone)]
pub struct Perceptron {
    pub weights: Vec<f64>,  // one weight per input
    pub bias: f64,          // single bias
}

impl Perceptron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }
    pub fn forward(&self, input: &[f64]) -> f64 {
        let mut sum: f64 = 0.0;
        
        for i in 0..self.weights.len() {
            sum += self.weights[i] * input[i];
        }
        sum + self.bias
    }
}
