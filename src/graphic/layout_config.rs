#[derive(Copy, Clone)]
pub struct LayoutConfig {
    pub layer_spacing: i32,
    pub node_spacing: i32,
    pub node_radius: i32,
    pub canvas_width: i32,
    pub canvas_height: i32,
}

impl LayoutConfig {
    pub fn new(canvas_width: i32, canvas_height: i32) -> Self {
        Self {
            layer_spacing: 200,
            node_spacing: 50,
            node_radius: 25,
            canvas_width,
            canvas_height,
        }
    }
    
    pub fn get_layer_x(&self, layer_number: i32) -> i32 {
        self.layer_spacing * layer_number + 50
    }
    
    pub fn get_layer_start_y(&self, num_nodes: i32) -> i32 {
        let total_layer_height = (num_nodes - 1) * (self.node_radius * 2 + self.node_spacing);
        (self.canvas_height - total_layer_height) / 2
    }
    
    pub fn get_node_y(&self, node_index: i32, layer_start_y: i32) -> i32 {
        node_index * (self.node_radius * 2 + self.node_spacing) + layer_start_y
    }
}