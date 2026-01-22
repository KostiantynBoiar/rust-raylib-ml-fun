mod ml;
mod graphic; 
mod data;

use ml::perceptron::Perceptron;
use ml::layer::Layer;
use ml::model::Model;
use graphic::window::Window;

fn main() {
    let window = Window::new(1920, 1080, String::from("Basic Model"));
    window.run();
}