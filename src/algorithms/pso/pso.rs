pub use crate::algorithms::pso::structs::swarm::Swarm;
pub use crate::structs::combination::Combination;
pub use crate::utils::read_data::read_data;

pub fn init(dataset: &str) {
    println!("Starting PSO algorithm for dataset: {}", dataset);

    // Read dataset and initialize combinations
    let combinations = read_data(dataset).expect("Failed to read dataset");
    println!("Dataset read successfully. Number of combinations: {}", combinations.len());

    // Hyperparameters for swarm
    let swarm_size = 20;
    let particle_size = combinations[0].combination.len(); // Size of each particle is the size of the combination

    // Hyperparameters for PSO
    let epochs = 100;

    let mut swarm = Swarm::new(swarm_size, particle_size);
    println!("Swarm initialized with {} particles of size {}", swarm_size, particle_size);

    // Initialize the best position and loss
    swarm.update_global_best();
    println!("Global best position: {:?}", swarm.best_position);
}