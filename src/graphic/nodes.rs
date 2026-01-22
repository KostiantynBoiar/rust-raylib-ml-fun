use crate::ml::layer::Layer;
use raylib::prelude::*;
use crate::graphic::connection::Connection;

pub struct Nodes{
    pub layer: Layer,
    pub radius: f32,
    pub color: Color,
}

impl Nodes{
    pub fn new(layer: Layer, radius: f32, color: Color) -> Self{
        Self { layer, radius, color }
    }
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        for (i, node) in self.layer.perceptrons.iter().enumerate() {
            let y = i as i32 * (self.radius as i32 + 50) + 50;
            let x = 50;
            d.draw_circle(x, y, self.radius, self.color);
            Connection::new((x, y), (x, y), node.weights[0]).draw(d);
        }
    }
}