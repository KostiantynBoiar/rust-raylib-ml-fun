use crate::ml::layer::Layer;
use raylib::prelude::*;
use crate::graphic::layout_config::LayoutConfig;

pub struct Nodes<'a> {
    pub layer: &'a Layer,
    pub radius: i32,
    pub color: Color,
    pub layer_number: i32,
    pub config: LayoutConfig,
}

impl<'a> Nodes<'a> {
    pub fn new(layer: &'a Layer, color: Color, layer_number: i32, config: LayoutConfig) -> Self {
        let radius = config.node_radius;
        Self { layer, radius, color, layer_number, config }
    }
    
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        for (i, node) in self.layer.perceptrons.iter().enumerate() {
            let x = self.config.get_layer_x(self.layer_number);
            let y = self.config.get_node_y(i as i32);
            d.draw_circle(x, y, self.radius as f32, self.color);
            self.draw_weights(d);
        }
    }
    
    pub fn draw_weights(&self, d: &mut RaylibDrawHandle) {
        for (i, node) in self.layer.perceptrons.iter().enumerate() {
            let x = self.config.get_layer_x(self.layer_number);
            let y = self.config.get_node_y(i as i32);
            
            let weight_text = format!("{:.2}", node.weights[0]);
            let font_size = 12;
            
            let text_width = d.measure_text(&weight_text, font_size);
            
            let text_x = x - text_width / 2;           // Center horizontally
            let text_y = y - font_size / 2;            // Center vertically
            
            d.draw_text(&weight_text, text_x, text_y, font_size, Color::WHITE);
        }
    }

    pub fn get_node_position(&self, node_index: usize) -> (i32, i32) {
        let x = self.config.get_layer_x(self.layer_number);
        let y = self.config.get_node_y(node_index as i32);
        (x, y)
    }
}