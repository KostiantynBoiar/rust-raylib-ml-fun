use crate::ml::layer::Layer;
use crate::ml::activation::Activation;

pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();

        for layer in &self.layers {
            current = layer.forward(&current);
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::perceptron::Perceptron;

    #[test]
    fn test_model_forward() {
        // Input: 2 values
        // Hidden layer: 3 neurons
        // Output layer: 1 neuron

        let hidden_layer = Layer::new(vec![
            Perceptron::new(vec![0.2, 0.3], 0.1),
            Perceptron::new(vec![0.4, 0.5], 0.2),
            Perceptron::new(vec![0.6, 0.7], 0.3),
        ], Activation::ReLU);

        let output_layer = Layer::new(vec![
            Perceptron::new(vec![0.1, 0.2, 0.3], 0.0),
        ], Activation::ReLU);

        let model = Model::new(vec![hidden_layer, output_layer]);

        let input = vec![1.0, 2.0];

        // Hidden layer:
        // Neuron 1: (0.2 * 1.0) + (0.3 * 2.0) + 0.1 = 0.9, ReLU = 0.9
        // Neuron 2: (0.4 * 1.0) + (0.5 * 2.0) + 0.2 = 1.6, ReLU = 1.6
        // Neuron 3: (0.6 * 1.0) + (0.7 * 2.0) + 0.3 = 2.3, ReLU = 2.3
        // Hidden output: [0.9, 1.6, 2.3]

        // Output layer:
        // Neuron 1: (0.1 * 0.9) + (0.2 * 1.6) + (0.3 * 2.3) + 0.0
        //         = 0.09 + 0.32 + 0.69
        //         = 1.1
        // ReLU = 1.1

        let output = model.forward(&input);
        assert!((output[0] - 1.1).abs() < 0.0001);
    }

    #[test]
    fn test_model_three_layers() {
        // 2 -> 4 -> 3 -> 1 architecture

        let layer_1 = Layer::new(vec![
            Perceptron::new(vec![1.0, 0.0], 0.0),
            Perceptron::new(vec![0.0, 1.0], 0.0),
            Perceptron::new(vec![0.5, 0.5], 0.0),
            Perceptron::new(vec![1.0, -1.0], 0.0),
        ], Activation::ReLU);

        let layer_2 = Layer::new(vec![
            Perceptron::new(vec![0.25, 0.25, 0.25, 0.25], 0.0),
            Perceptron::new(vec![1.0, 0.0, 0.0, 0.0], 0.0),
            Perceptron::new(vec![0.0, 0.0, 0.0, 1.0], 0.0),
        ], Activation::ReLU);

        let layer_3 = Layer::new(vec![
            Perceptron::new(vec![1.0, 1.0, 1.0], 0.0),
        ], Activation::ReLU);

        let model = Model::new(vec![layer_1, layer_2, layer_3]);

        let input = vec![4.0, 2.0];

        // Layer 1: [4.0, 2.0, 3.0, 2.0]
        // Layer 2: [2.75, 4.0, 2.0]
        // Layer 3: [8.75]

        let output = model.forward(&input);
        assert!((output[0] - 8.75).abs() < 0.0001);
    }
}