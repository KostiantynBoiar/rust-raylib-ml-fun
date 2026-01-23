mod ml;
mod graphic; 
mod data;

use graphic::window::Window;

fn main() {
    let window = Window::new(1920, 1080, String::from("Basic Model"));
    window.run();
}