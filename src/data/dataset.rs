use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::prelude::*;

pub struct Dataset{
    pub train_data: Vec<(Vec<f64>, Vec<f64>)>,
    pub test_data: Vec<(Vec<f64>, Vec<f64>)>
}

impl Dataset{
    pub fn new(train_data: Vec<(Vec<f64>, Vec<f64>)>, test_data: Vec<(Vec<f64>, Vec<f64>)>) -> Self {
        Self { train_data, test_data }
    }
    pub fn load_data(path: &str, split_ratio: f64) -> Result<Self, Box<dyn Error>>{
        let file = File::open(path).expect("Failed to open file");
        let reader = BufReader::new(file);
        
        let mut all_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();


        for line in reader.lines(){
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }
            let values: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();

            let features = values[0..57].to_vec();
            let target = vec![values[57]];
            all_data.push((features, target));
            shuffle_data(&mut all_data);
        }
        let (train_data, test_data) = split_data(all_data, split_ratio);
        Ok(Dataset::new(train_data, test_data))
    }

}
//TODO: move to utils both functions
fn shuffle_data(data: &mut Vec<(Vec<f64>, Vec<f64>)>) {
    let mut rng = rand::rng();
    data.shuffle(&mut rng);
}
fn split_data(
    data: Vec<(Vec<f64>, Vec<f64>)>,
    split_ratio: f64,
) -> (Vec<(Vec<f64>, Vec<f64>)>, Vec<(Vec<f64>, Vec<f64>)>) {
    let split_idx = (data.len() as f64 * split_ratio) as usize;
    let train = data[0..split_idx].to_vec();
    let test = data[split_idx..].to_vec();
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_spambase() {
        let dataset = Dataset::load_data("spambase/spambase.data", 0.8).unwrap();

        println!("Train: {}", dataset.train_data.len());
        println!("Test: {}", dataset.test_data.len());

        // Check dimensions
        assert_eq!(dataset.train_data[0].0.len(), 57);
        assert_eq!(dataset.train_data[0].1.len(), 1);
    }
}