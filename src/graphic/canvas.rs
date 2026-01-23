use raylib::prelude::*;
use rand::prelude::*;
use crate::ml::perceptron::Perceptron;
use crate::ml::layer::Layer;
use crate::graphic::model_visualisation::ModelVisualisation;
use crate::ml::model::Model;
use crate::ml::activation::Activation;
use crate::data::dataset::Dataset;

pub struct Canvas {
    width: i32,
    height: i32,
    color: Color,
    model_visualisation: ModelVisualisation,
    dataset: Dataset,
    current_epoch: i32,
    current_loss: f64,
    is_training: bool
}

impl Canvas {
    pub fn new(width: i32, height: i32, color: Color) -> Self {

        let (model, dataset) = create_spam_classifier();
        let model_visualisation = ModelVisualisation::new(model);
        
        Self { 
            width, 
            height, 
            color,
            model_visualisation,
            dataset,
            current_epoch: 0,
            current_loss: 0.0,
            is_training: true,
        }
    }
    pub fn update(&mut self){
        if self.is_training{
            self.current_loss = self.model_visualisation.model.train_epoch(&self.dataset.train_data, 0.01);
            self.current_epoch += 1;
            if self.current_epoch % 10 == 0{
                println!("Epoch {}: loss = {:.4}", self.current_epoch, self.current_loss);
            }
        }
        if self.current_epoch >= 100{
            self.is_training = false;
        }
    }
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        self.model_visualisation.draw(d);
    }
}

fn create_spam_classifier() -> (Model, Dataset) {
    let mut dataset = Dataset::load_data("spambase/spambase.data", 0.8).unwrap();
    dataset.normalize();

    let hidden1 = create_layer(57, 32, Activation::ReLU);
    let hidden2 = create_layer(32, 16, Activation::ReLU);
    let output = create_layer(16, 1, Activation::Sigmoid);

    let model = Model::new(vec![hidden1, hidden2, output]);

    (model, dataset)
}
fn create_layer(num_inputs: usize, num_neurons: usize, activation: Activation) -> Layer {
    let mut rng = rand::rng();
    let mut perceptrons = Vec::new();

    for _ in 0..num_neurons {
        let weights: Vec<f64> = (0..num_inputs)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let bias = rng.random_range(-0.1..0.1);
        
        perceptrons.push(Perceptron::new(weights, bias));
    }

    Layer::new(perceptrons, activation)
}