use crate::ml::perceptron::Perceptron;
use crate::ml::activation::Activation;

pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
}

impl Layer {
    pub fn new(perceptrons: Vec<Perceptron>) -> Self {
        Self { perceptrons }
    }
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = Vec::new();
        for perceptron in &self.perceptrons {
            let weighted_sum = perceptron.forward(input);
            let activated = Activation::ReLU.activate(weighted_sum);
            output.push(activated);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(vec![Perceptron::new(vec![0.0], 0.0)]);
        let input = vec![0.0, 0.0];
        let output = layer.forward(&input);
        assert_eq!(output, vec![0.0]);
    }
    #[test]
    fn test_layer_forward_with_multiple_perceptrons() {
        let perceptron = Perceptron::new(vec![0.5, -0.3], 0.1);
        let layer = Layer::new(vec![perceptron]);
        let input = vec![2.0, 4.0];
        let output = layer.forward(&input);
        assert_eq!(output, vec![0.0]);
    }

    #[test]
    fn test_single_perceptron_positive_output() {
        let perceptron = Perceptron::new(vec![0.5, 0.3], 0.1);
        let layer = Layer::new(vec![perceptron]);

        let input = vec![2.0, 4.0];

        // Manual calculation:
        // weighted_sum = (0.5 * 2.0) + (0.3 * 4.0) + 0.1
        //              = 1.0 + 1.2 + 0.1
        //              = 2.3
        // ReLU(2.3) = 2.3  (positive values stay the same)

        let output = layer.forward(&input);
        assert!((output[0] - 2.3).abs() < 1e-10);
    }


}