use raylib::prelude::*;
use rand::prelude::*;
use crate::ml::perceptron::Perceptron;
use crate::ml::layer::Layer;
use crate::graphic::model_visualisation::ModelVisualisation;
use crate::ml::model::Model;
use crate::ml::activation::Activation;

pub struct Canvas {
    width: i32,
    height: i32,
    color: Color,
    model_visualisation: ModelVisualisation,  // Store it here!
}

impl Canvas {
    pub fn new(width: i32, height: i32, color: Color) -> Self {

        let model = Model::new(vec![
            layer_generator(2), 
            layer_generator(6), 
            layer_generator(3),
            layer_generator(1)
        ]);
        let model_visualisation = ModelVisualisation::new(model);
        
        Self { 
            width, 
            height, 
            color,
            model_visualisation 
        }
    }
    
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        self.model_visualisation.draw(d);
    }
}

fn layer_generator(amount: i32) -> Layer {
    let mut rng = rand::rng();
    let mut perceptrons = Vec::new();
    for _ in 0..amount {
        perceptrons.push(Perceptron::new(
            vec![rng.gen_range(-1.0..1.0)], 
            rng.gen_range(-1.0..1.0)
        ));
    }
    Layer::new(perceptrons, Activation::ReLU)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percepton_generator() {
        let perceptrons = percepton_generator(3);
        assert_eq!(perceptrons.len(), 3);
    }
}