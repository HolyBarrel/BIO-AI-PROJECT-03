use crate::structs::combination::Combination; // Import the Combination struct from the combination module
use crate::utils::read_data;
use rand::prelude::*;

pub fn create_individual(gene_length: usize, data: &Vec<Combination>) -> Combination {
    let mut activated_columns = vec![false; gene_length];
    let mut rng = rand::rng();

    let num_columns = 9;
    for i in 0..num_columns {
        let random_value: f64 = rng.random_range(0.0..1.0);
        if random_value < 0.5 {
            activated_columns[i] = true; 
        } else {
            activated_columns[i] = false; 
        }
    } 

    for i in data.iter() {
        if i.combination == activated_columns {
            return i.clone();
        }
    }

    // If no matching combination is found, return a new individual with the activated columns
    Combination {
        combination: activated_columns,
        loss: 0.0, // Initialize loss to 0.0 or some other value
    }

}

pub fn init_population(gene_length: usize, population_size: usize) -> Vec<Combination> {
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; // Path to the CSV file
    let data = read_data::read_data(file_path).unwrap(); // Read data from the file and unwrap the Result
    let mut population: Vec<Combination> = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let combination = create_individual(gene_length, &data);
        population.push(combination);
    }
    population
}


pub fn init() {
    // Initialize the multi-objective optimization algorithm
    let gene_length = 9;
    let population = init_population(gene_length, 50); // Initialize the population

    // Print the initialized population for debugging
    for individual in &population {
        println!("{:?}", individual);
    }
}

