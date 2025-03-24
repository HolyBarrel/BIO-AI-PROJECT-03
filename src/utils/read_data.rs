use std::error::Error;
use csv::ReaderBuilder;

use crate::structs::combination::Combination;

pub fn read_data(file_path: &str) -> Result<Vec<Combination>, Box<dyn Error>> {
    let mut combinations: Vec<Combination> = Vec::new();
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
    let headers = reader.headers()?;
    let loss_index = headers.iter().position(|h| h == "Loss")
        .ok_or("The 'Loss' column was not found in the header")?;
    for result in reader.records() {
        let record = result?;
        let loss_str = record.get(loss_index)
            .ok_or("Missing loss value in record")?;
        let loss = loss_str.parse::<f64>()?;
        let mut combination: Vec<bool> = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i != loss_index {
                let int_val = value.parse::<i32>()?;
                combination.push(int_val == 1);
            }
        }
        
        combinations.push(Combination { combination, loss });
    }

    Ok(combinations)
}
