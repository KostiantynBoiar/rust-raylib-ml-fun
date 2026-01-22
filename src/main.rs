mod ml;
mod graphic; 

use ml::perceptron::Perceptron;
use ml::layer::Layer;
use ml::model::Model;
use graphic::window::Window;

fn main() {
    let window = Window::new(640, 480, String::from("Basic Model"));
    window.run();
}