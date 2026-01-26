use raylib::prelude::*;
use rand::prelude::*;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use crate::ml::perceptron::Perceptron;
use crate::ml::layer::Layer;
use crate::graphic::model_visualisation::ModelVisualisation;
use crate::graphic::camera::Camera;
use crate::ml::model::Model;
use crate::ml::activation::Activation;
use crate::data::dataset::Dataset;

struct TrainingState {
    model: Arc<Mutex<Model>>,
    dataset: Arc<Dataset>,
    epoch: Arc<AtomicU32>,
    loss: Arc<Mutex<f64>>,
    is_running: Arc<AtomicBool>,
}

pub struct Canvas {
    width: i32,
    height: i32,
    color: Color,
    model_visualisation: ModelVisualisation,
    training_state: TrainingState,
    training_thread: Option<JoinHandle<()>>,
    last_update_epoch: u32,
    update_interval: u32,
    camera: Camera,
}

impl Canvas {
    pub fn new(width: i32, height: i32, color: Color) -> Self {
        let (model, dataset) = create_spam_classifier();
        let model_visualisation = ModelVisualisation::new(model.clone());
        
        let camera = Camera::new(width, height);
        
        let shared_model = Arc::new(Mutex::new(model));
        let shared_dataset = Arc::new(dataset);
        let shared_epoch = Arc::new(AtomicU32::new(0));
        let shared_loss = Arc::new(Mutex::new(0.0));
        let shared_running = Arc::new(AtomicBool::new(true));
        
        let training_state = TrainingState {
            model: shared_model.clone(),
            dataset: shared_dataset.clone(),
            epoch: shared_epoch.clone(),
            loss: shared_loss.clone(),
            is_running: shared_running.clone(),
        };
        
        let thread_model = shared_model.clone();
        let thread_dataset = shared_dataset.clone();
        let thread_epoch = shared_epoch.clone();
        let thread_loss = shared_loss.clone();
        let thread_running = shared_running.clone();
        
        let training_thread = thread::spawn(move || {
            Self::training_loop(thread_running, thread_model, thread_dataset, thread_epoch, thread_loss);
        });
        
        Self { 
            width, 
            height, 
            color,
            model_visualisation,
            training_state,
            training_thread: Some(training_thread),
            last_update_epoch: 0,
            update_interval: 10,
            camera,
        }
    }

    fn training_loop(
        thread_running: Arc<AtomicBool>,
        thread_model: Arc<Mutex<Model>>,
        thread_dataset: Arc<Dataset>,
        thread_epoch: Arc<AtomicU32>,
        thread_loss: Arc<Mutex<f64>>,
    ) {
        let mut current_epoch = 0u32;
        
        while thread_running.load(Ordering::Relaxed) && current_epoch < 100 {
            let loss = {
                let mut model = thread_model.lock().unwrap();
                model.train_epoch(&thread_dataset.train_data, 0.01)
            };
            
            current_epoch += 1;
            thread_epoch.store(current_epoch, Ordering::Relaxed);
            *thread_loss.lock().unwrap() = loss;
            
            if current_epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.4}", current_epoch, loss);
            }
            
            thread::sleep(Duration::from_millis(1));
        }
        
        thread_running.store(false, Ordering::Relaxed);
        println!("Training complete!");
    }

    pub fn update(&mut self, rl: &RaylibHandle){
        let current_epoch = self.training_state.epoch.load(Ordering::Relaxed);
        
        if current_epoch >= self.last_update_epoch + self.update_interval {
            let model = self.training_state.model.lock().unwrap().clone();
            self.model_visualisation = ModelVisualisation::new(model);
            self.last_update_epoch = current_epoch;
        }
        
        if rl.is_key_pressed(KeyboardKey::KEY_SPACE) {
            let current_state = self.training_state.is_running.load(Ordering::Relaxed);
            self.training_state.is_running.store(!current_state, Ordering::Relaxed);
            if !current_state {
                println!("Training resumed");
            } else {
                println!("Training paused");
            }
        }
        
        self.camera.handle_input(rl);
    }
    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        {
            let mut camera_mode = d.begin_mode2D(self.camera.as_camera2d());
            self.model_visualisation.draw(&mut camera_mode);
        }
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {
        self.training_state.is_running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.training_thread.take() {
            let _ = handle.join();
            println!("Training thread stopped");
        }
    }
}

fn create_spam_classifier() -> (Model, Dataset) {
    let mut dataset = Dataset::load_data("spambase/spambase.data", 0.8).unwrap();
    dataset.normalize();

    let hidden1 = create_layer(57, 32, Activation::ReLU);
    let hidden2 = create_layer(32, 16, Activation::ReLU);
    let output = create_layer(16, 1, Activation::Sigmoid);

    let model = Model::new(vec![hidden1, hidden2, output]);

    (model, dataset)
}
fn create_layer(num_inputs: usize, num_neurons: usize, activation: Activation) -> Layer {
    let mut rng = rand::rng();
    let mut perceptrons = Vec::new();

    for _ in 0..num_neurons {
        let weights: Vec<f64> = (0..num_inputs)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let bias = rng.random_range(-0.1..0.1);
        
        perceptrons.push(Perceptron::new(weights, bias));
    }

    Layer::new(perceptrons, activation)
}