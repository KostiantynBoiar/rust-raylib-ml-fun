use raylib::prelude::*;
use crate::graphic::nodes::Nodes;

pub struct Connection<'a> {
    pub start_layer: &'a Nodes<'a>,
    pub end_layer: &'a Nodes<'a>,
}

impl<'a> Connection<'a> {
    pub fn new(start_layer: &'a Nodes<'a>, end_layer: &'a Nodes<'a>) -> Self {
        Self { start_layer, end_layer }
    }
    
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        for (i, _node) in self.start_layer.layer.perceptrons.iter().enumerate() {
            for (j, _node2) in self.end_layer.layer.perceptrons.iter().enumerate() {
                let (start_x, start_y) = self.start_layer.get_node_position(i);
                let (end_x, end_y) = self.end_layer.get_node_position(j);
                d.draw_line(start_x, start_y, end_x, end_y, Color::RED);
            }
        }
    }
}