#[derive(Copy, Clone)]  // Add this line!
pub struct LayoutConfig {
    pub layer_spacing: i32,
    pub node_spacing: i32,
    pub node_start_y: i32,
    pub node_radius: i32,
}

impl LayoutConfig {
    pub fn default() -> Self {
        Self {
            layer_spacing: 200,
            node_spacing: 50,
            node_start_y: 150,
            node_radius: 10,
        }
    }
    
    pub fn get_layer_x(&self, layer_number: i32) -> i32 {
        self.layer_spacing * layer_number
    }
    
    pub fn get_node_y(&self, node_index: i32) -> i32 {
        node_index * (self.node_radius + self.node_spacing) + self.node_start_y
    }
}