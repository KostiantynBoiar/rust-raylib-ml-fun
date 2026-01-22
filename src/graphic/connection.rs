use raylib::prelude::*;

pub struct Connection {
    pub start_pos: (i32, i32),  // (x, y) of source node
    pub end_pos: (i32, i32),    // (x, y) of target node
    pub weight: f64,            // connection weight
    pub color: Color,
}

impl Connection {
    pub fn new(start_pos: (i32, i32), end_pos: (i32, i32), weight: f64) -> Self {
        let color = if weight > 0.0 {
            Color::GREEN
        } else {
            Color::RED
        };
        
        Self { start_pos, end_pos, weight, color }
    }
    
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        let thickness = (self.weight.abs() * 3.0).max(1.0).min(5.0) as f32;
        d.draw_line_ex(
            Vector2::new(self.start_pos.0 as f32, self.start_pos.1 as f32),
            Vector2::new(self.end_pos.0 as f32, self.end_pos.1 as f32),
            thickness,
            self.color,
        );
    }
}