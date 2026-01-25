use raylib::prelude::*;
use crate::ml::model::Model;
use crate::graphic::nodes::Nodes;
use crate::graphic::connection::Connection;
use crate::graphic::layout_config::LayoutConfig;

pub struct ModelVisualisation {
    pub model: Model,
}

impl ModelVisualisation {
    pub fn new(model: Model) -> Self {
        Self { model }
    }
    
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        let canvas_width = d.get_screen_width();
        let canvas_height = d.get_screen_height();
        let config = LayoutConfig::new(canvas_width, canvas_height);
        
        let mut node_layers: Vec<Nodes> = Vec::new();
        
        for (layer_number, layer) in self.model.layers.iter().enumerate() {
            let nodes = Nodes::new(layer, Color::RED, layer_number as i32, config);
            node_layers.push(nodes);
        }
        
        // Draw connections
        for i in 0..node_layers.len() - 1 {
            let start_layer = &node_layers[i];
            let end_layer = &node_layers[i + 1];
            let connection = Connection::new(start_layer, end_layer);
            connection.draw(d);
        }
        
        // Draw nodes
        for nodes in &node_layers {
            nodes.draw(d);
        }
    }
}