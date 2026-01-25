use raylib::prelude::*;

pub struct Camera {
    camera: Camera2D,
    pan_speed: f32,
    zoom_speed: f32,
    min_zoom: f32,
    max_zoom: f32,
}

impl Camera {
    pub fn new(screen_width: i32, screen_height: i32) -> Self {
        Self {
            camera: Camera2D {
                target: Vector2::new(0.0, 0.0),
                offset: Vector2::new(screen_width as f32 / 2.0, screen_height as f32 / 2.0),
                rotation: 0.0,
                zoom: 1.0,
            },
            pan_speed: 10.0,
            zoom_speed: 0.1,
            min_zoom: 0.1,
            max_zoom: 3.0,
        }
    }

    pub fn with_settings(
        screen_width: i32,
        screen_height: i32,
        pan_speed: f32,
        zoom_speed: f32,
        min_zoom: f32,
        max_zoom: f32,
    ) -> Self {
        Self {
            camera: Camera2D {
                target: Vector2::new(0.0, 0.0),
                offset: Vector2::new(screen_width as f32 / 2.0, screen_height as f32 / 2.0),
                rotation: 0.0,
                zoom: 1.0,
            },
            pan_speed,
            zoom_speed,
            min_zoom,
            max_zoom,
        }
    }

    pub fn as_camera2d(&self) -> Camera2D {
        self.camera
    }

    pub fn handle_pan_input(&mut self, rl: &RaylibHandle) {
        if rl.is_key_down(KeyboardKey::KEY_RIGHT) {
            self.camera.target.x += self.pan_speed;
        }
        if rl.is_key_down(KeyboardKey::KEY_LEFT) {
            self.camera.target.x -= self.pan_speed;
        }
        if rl.is_key_down(KeyboardKey::KEY_DOWN) {
            self.camera.target.y += self.pan_speed;
        }
        if rl.is_key_down(KeyboardKey::KEY_UP) {
            self.camera.target.y -= self.pan_speed;
        }
    }

    pub fn handle_zoom_input(&mut self, rl: &RaylibHandle) {
        let mouse_pos = rl.get_mouse_position();

        if rl.is_key_pressed(KeyboardKey::KEY_EQUAL) || rl.is_key_pressed(KeyboardKey::KEY_KP_ADD) {
            self.zoom_in_towards(mouse_pos, rl);
        }

        if rl.is_key_pressed(KeyboardKey::KEY_MINUS) || rl.is_key_pressed(KeyboardKey::KEY_KP_SUBTRACT) {
            self.zoom_out_towards(mouse_pos, rl);
        }
    }

    pub fn zoom_in_towards(&mut self, screen_pos: Vector2, rl: &RaylibHandle) {
        let mouse_world_pos = rl.get_screen_to_world2D(screen_pos, self.camera);
        self.camera.zoom += self.zoom_speed;
        self.camera.zoom = self.camera.zoom.min(self.max_zoom);
        let new_mouse_world_pos = rl.get_screen_to_world2D(screen_pos, self.camera);
        self.camera.target.x += mouse_world_pos.x - new_mouse_world_pos.x;
        self.camera.target.y += mouse_world_pos.y - new_mouse_world_pos.y;
    }

    pub fn zoom_out_towards(&mut self, screen_pos: Vector2, rl: &RaylibHandle) {
        let mouse_world_pos = rl.get_screen_to_world2D(screen_pos, self.camera);
        self.camera.zoom -= self.zoom_speed;
        self.camera.zoom = self.camera.zoom.max(self.min_zoom);
        let new_mouse_world_pos = rl.get_screen_to_world2D(screen_pos, self.camera);
        self.camera.target.x += mouse_world_pos.x - new_mouse_world_pos.x;
        self.camera.target.y += mouse_world_pos.y - new_mouse_world_pos.y;
    }

    pub fn handle_input(&mut self, rl: &RaylibHandle) {
        self.handle_pan_input(rl);
        self.handle_zoom_input(rl);
    }

    pub fn screen_to_world(&self, screen_pos: Vector2, rl: &RaylibHandle) -> Vector2 {
        rl.get_screen_to_world2D(screen_pos, self.camera)
    }

    pub fn world_to_screen(&self, world_pos: Vector2, rl: &RaylibHandle) -> Vector2 {
        rl.get_world_to_screen2D(world_pos, self.camera)
    }

    pub fn target(&self) -> Vector2 {
        self.camera.target
    }

    pub fn set_target(&mut self, target: Vector2) {
        self.camera.target = target;
    }

    pub fn zoom(&self) -> f32 {
        self.camera.zoom
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.camera.zoom = zoom.clamp(self.min_zoom, self.max_zoom);
    }

    pub fn reset(&mut self) {
        self.camera.target = Vector2::new(0.0, 0.0);
        self.camera.zoom = 1.0;
    }

    pub fn pan_speed(&self) -> f32 {
        self.pan_speed
    }

    pub fn set_pan_speed(&mut self, speed: f32) {
        self.pan_speed = speed;
    }

    pub fn zoom_speed(&self) -> f32 {
        self.zoom_speed
    }

    pub fn set_zoom_speed(&mut self, speed: f32) {
        self.zoom_speed = speed;
    }
}
