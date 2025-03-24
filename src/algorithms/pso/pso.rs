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
    let w = 0.5; // Inertia weight
    let c1 = 1.0; // Cognitive weight
    let c2 = 1.0; // Social weight
    let epsilon = 0.1; // Small value to avoid division by zero
    let epochs = 1000;

    let mut swarm = Swarm::new(swarm_size, particle_size);
    println!("Swarm initialized with {} particles of size {}", swarm_size, particle_size);

    // Main loop for PSO
    for epoch in 0..epochs {
        println!("Epoch {}: Starting particle updates", epoch);
        swarm.update_particles(w, c1, c2, epsilon, &combinations);
        swarm.update_global_best();
        println!("Epoch {}: Global best loss: {}", epoch, swarm.best_loss);
    }
    println!("PSO algorithm completed. Final global best loss: {}", swarm.best_loss);
    println!("Final global best position: {:?}", swarm.best_position);
}