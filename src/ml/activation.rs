pub enum Activation {
    ReLU,
}

impl Activation {
    pub fn activate(&self, input: f64) -> f64 {
        match self {
            Activation::ReLU => input.max(0.0),
        }
    }
}
