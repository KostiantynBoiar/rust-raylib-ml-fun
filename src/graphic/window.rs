use raylib::prelude::*;
use crate::graphic::canvas::Canvas;

pub struct Window{
    width: i32,
    height: i32,
    title: String,
}

impl Window{
    pub fn new(width: i32, height: i32, title: String) -> Self {
        Self { width, height, title }
    }
}

impl Window{
    pub fn run(&self) {
        let (mut rl, thread) = raylib::init()
            .size(self.width, self.height)
            .title(&self.title)
            .build();

        let mut canvas = Canvas::new(self.width, self.height, Color::BLACK);

        while !rl.window_should_close() {
            canvas.update();
            let mut d = rl.begin_drawing(&thread);
            d.clear_background(Color::WHITE);
            d.draw_rectangle(0, 0, self.width, self.height, Color::WHITE);
            canvas.draw(&mut d);
        }
    }
}