use crate::ml::perceptron::Perceptron;
use crate::ml::activation::Activation;

pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub activation: Activation,
    // Cached values from forward pass (needed for backprop)
    last_input: Vec<f64>,
    last_weighted_sums: Vec<f64>,
}

impl Layer {
    pub fn new(perceptrons: Vec<Perceptron>, activation: Activation) -> Self {
        Self { perceptrons, activation, last_input: Vec::new(), last_weighted_sums: Vec::new() }
    }
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = Vec::new();
        self.last_input = input.to_vec();
        self.last_weighted_sums = Vec::new(); 
        for perceptron in &self.perceptrons {
            let weighted_sum = perceptron.forward(input);
            self.last_weighted_sums.push(weighted_sum);
            let activated = self.activation.activate(weighted_sum);
            output.push(activated);
        }
        output
    }
    pub fn backward(&mut self, output_gradients: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut input_gradients = vec![0.0; self.last_input.len()];
        for i in 0..self.perceptrons.len() {
            let activation_derivative = self.activation.derivative(self.last_weighted_sums[i]);
            let delta = output_gradients[i] * activation_derivative;
            for j in 0..self.perceptrons[i].weights.len() {
                input_gradients[j] += delta * self.perceptrons[i].weights[j];
                self.perceptrons[i].weights[j] -= learning_rate * delta * self.last_input[j];
            }
            self.perceptrons[i].bias -= learning_rate * delta;
        }
        input_gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(vec![Perceptron::new(vec![0.0], 0.0)], Activation::ReLU);
        let input = vec![0.0, 0.0];
        let output = layer.forward(&input);
        assert_eq!(output, vec![0.0]);
    }

    #[test]
    fn test_layer_forward_with_multiple_perceptrons() {
        let perceptron = Perceptron::new(vec![0.5, -0.3], 0.1);
        let mut layer = Layer::new(vec![perceptron], Activation::ReLU);
        let input = vec![2.0, 4.0];
        let output = layer.forward(&input);
        assert_eq!(output, vec![0.0]);
    }

    #[test]
    fn test_single_perceptron_positive_output() {
        let perceptron = Perceptron::new(vec![0.5, 0.3], 0.1);
        let mut layer = Layer::new(vec![perceptron], Activation::ReLU);
        let input = vec![2.0, 4.0];
        let output = layer.forward(&input);
        assert!((output[0] - 2.3).abs() < 1e-10);
    }

    #[test]
    fn test_backward_updates_weights() {
        // Single perceptron: 2 inputs, 1 output
        let perceptron = Perceptron::new(vec![0.5, 0.3], 0.1);
        let mut layer = Layer::new(vec![perceptron], Activation::Sigmoid);

        let input = vec![1.0, 2.0];
        let learning_rate = 0.1;

        // Forward pass
        let output = layer.forward(&input);

        // Save old weights
        let old_weight_0 = layer.perceptrons[0].weights[0];
        let old_weight_1 = layer.perceptrons[0].weights[1];
        let old_bias = layer.perceptrons[0].bias;

        // Backward pass with gradient of 1.0
        let output_gradients = vec![1.0];
        let _input_gradients = layer.backward(&output_gradients, learning_rate);

        // Weights should have changed
        assert_ne!(layer.perceptrons[0].weights[0], old_weight_0);
        assert_ne!(layer.perceptrons[0].weights[1], old_weight_1);
        assert_ne!(layer.perceptrons[0].bias, old_bias);

        println!("Output: {:?}", output);
        println!("Weight 0: {} -> {}", old_weight_0, layer.perceptrons[0].weights[0]);
        println!("Weight 1: {} -> {}", old_weight_1, layer.perceptrons[0].weights[1]);
        println!("Bias: {} -> {}", old_bias, layer.perceptrons[0].bias);
    }

    #[test]
    fn test_backward_manual_calculation() {
        // Simple case: 1 input, 1 output, ReLU
        let perceptron = Perceptron::new(vec![0.5], 0.0);
        let mut layer = Layer::new(vec![perceptron], Activation::ReLU);

        let input = vec![2.0];
        let learning_rate = 0.1;

        // Forward: weighted_sum = 0.5 * 2.0 = 1.0, ReLU(1.0) = 1.0
        let output = layer.forward(&input);
        assert!((output[0] - 1.0).abs() < 1e-10);

        // Backward with gradient = 0.5
        // delta = 0.5 * ReLU_derivative(1.0) = 0.5 * 1.0 = 0.5
        // weight_update = learning_rate * delta * input = 0.1 * 0.5 * 2.0 = 0.1
        // new_weight = 0.5 - 0.1 = 0.4
        let output_gradients = vec![0.5];
        layer.backward(&output_gradients, learning_rate);

        assert!((layer.perceptrons[0].weights[0] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_backward_returns_input_gradients() {
        // 2 inputs, 2 neurons
        let p1 = Perceptron::new(vec![0.1, 0.2], 0.0);
        let p2 = Perceptron::new(vec![0.3, 0.4], 0.0);
        let mut layer = Layer::new(vec![p1, p2], Activation::ReLU);

        let input = vec![1.0, 1.0];
        layer.forward(&input);

        let output_gradients = vec![1.0, 1.0];
        let input_gradients = layer.backward(&output_gradients, 0.1);

        // Should return gradients for each input
        assert_eq!(input_gradients.len(), 2);

        // input_gradient[0] = delta1 * weight1_0 + delta2 * weight2_0
        // With ReLU derivative = 1 (positive sums): = 1.0 * 0.1 + 1.0 * 0.3 = 0.4
        assert!((input_gradients[0] - 0.4).abs() < 1e-10);

        // input_gradient[1] = 1.0 * 0.2 + 1.0 * 0.4 = 0.6
        assert!((input_gradients[1] - 0.6).abs() < 1e-10);
    }
}