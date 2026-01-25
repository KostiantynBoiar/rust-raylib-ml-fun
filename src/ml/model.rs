use crate::ml::layer::Layer;
use crate::ml::loss::Loss;

#[derive(Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn train(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {

        let output = self.forward(input);

        let mut loss = 0.0;
        for i in 0..output.len() {
            loss += Loss::SumSquaredError.calculate(output[i], target[i]);
        }

        let mut gradients = Vec::new();
        for i in 0..output.len() {
            gradients.push(Loss::SumSquaredError.derivative(output[i], target[i]));
        }

        for layer in self.layers.iter_mut().rev() {
            gradients = layer.backward(&gradients, learning_rate);
        }

        loss
    }

    pub fn train_epoch(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
        learning_rate: f64,
    ) -> f64 {
        let mut total_loss = 0.0;

        for (input, target) in data {
            total_loss += self.train(input, target, learning_rate);
        }

        total_loss / data.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::perceptron::Perceptron;
    use crate::ml::activation::Activation;

    #[test]
    fn test_model_forward() {
        let hidden_layer = Layer::new(vec![
            Perceptron::new(vec![0.2, 0.3], 0.1),
            Perceptron::new(vec![0.4, 0.5], 0.2),
            Perceptron::new(vec![0.6, 0.7], 0.3),
        ], Activation::ReLU);

        let output_layer = Layer::new(vec![
            Perceptron::new(vec![0.1, 0.2, 0.3], 0.0),
        ], Activation::ReLU);

        let mut model = Model::new(vec![hidden_layer, output_layer]);
        let input = vec![1.0, 2.0];

        let output = model.forward(&input);
        assert!((output[0] - 1.1).abs() < 0.0001);
    }

    #[test]
    fn test_model_three_layers() {
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

        let mut model = Model::new(vec![layer_1, layer_2, layer_3]);
        let input = vec![4.0, 2.0];

        let output = model.forward(&input);
        assert!((output[0] - 8.75).abs() < 0.0001);
    }

    #[test]
    fn test_train_reduces_loss() {
        // Simple network: learn to output 1.0 when input is 1.0
        let layer = Layer::new(vec![
            Perceptron::new(vec![0.5], 0.0),
        ], Activation::Sigmoid);

        let mut model = Model::new(vec![layer]);

        let input = vec![1.0];
        let target = vec![1.0];
        let learning_rate = 0.5;

        let loss_before = model.train(&input, &target, learning_rate);
        
        // Train a few more times
        for _ in 0..100 {
            model.train(&input, &target, learning_rate);
        }

        let output = model.forward(&input);
        let loss_after = Loss::SumSquaredError.calculate(output[0], target[0]);

        println!("Loss before: {}", loss_before);
        println!("Loss after: {}", loss_after);
        println!("Output: {}", output[0]);

        assert!(loss_after < loss_before);
    }

    #[test]
    fn test_learn_xor() {
        // XOR requires hidden layer
        let hidden = Layer::new(vec![
            Perceptron::new(vec![0.5, 0.5], -0.2),
            Perceptron::new(vec![0.5, 0.5], -0.7),
        ], Activation::Sigmoid);

        let output = Layer::new(vec![
            Perceptron::new(vec![0.5, -0.5], 0.0),
        ], Activation::Sigmoid);

        let mut model = Model::new(vec![hidden, output]);

        let xor_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        // Train
        let learning_rate = 1.0;
        for epoch in 0..1000 {
            let loss = model.train_epoch(&xor_data, learning_rate);
            
            if epoch % 200 == 0 {
                println!("Epoch {}: loss = {:.4}", epoch, loss);
            }
        }

        // Test
        println!("\nResults:");
        for (input, target) in &xor_data {
            let output = model.forward(input);
            let predicted = if output[0] > 0.5 { 1 } else { 0 };
            println!(
                "{:?} -> {:.3} (predicted: {}, expected: {})",
                input, output[0], predicted, target[0]
            );
        }
    }
}