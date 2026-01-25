#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_activate() {
        let relu = Activation::ReLU;
        
        assert_eq!(relu.activate(5.0), 5.0);
        assert_eq!(relu.activate(0.0), 0.0);
        assert_eq!(relu.activate(-3.0), 0.0);
    }

    #[test]
    fn test_relu_derivative() {
        let relu = Activation::ReLU;
        
        assert_eq!(relu.derivative(5.0), 1.0);
        assert_eq!(relu.derivative(-3.0), 0.0);
    }

    #[test]
    fn test_sigmoid_activate() {
        let sigmoid = Activation::Sigmoid;
        
        // sigmoid(0) = 0.5
        assert!((sigmoid.activate(0.0) - 0.5).abs() < 1e-10);
        
        // Large positive → close to 1
        assert!(sigmoid.activate(10.0) > 0.99);
        
        // Large negative → close to 0
        assert!(sigmoid.activate(-10.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sigmoid = Activation::Sigmoid;
        
        // derivative at x=0: s(0) * (1 - s(0)) = 0.5 * 0.5 = 0.25
        assert!((sigmoid.derivative(0.0) - 0.25).abs() < 1e-10);
        
        // Derivative is always positive and max at x=0
        assert!(sigmoid.derivative(0.0) > sigmoid.derivative(2.0));
        assert!(sigmoid.derivative(0.0) > sigmoid.derivative(-2.0));
    }
}