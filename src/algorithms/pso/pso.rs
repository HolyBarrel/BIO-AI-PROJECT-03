pub use crate::algorithms::pso::structs::swarm::Swarm;
pub use crate::algorithms::pso::structs::pso_mode::UpdateMode;
pub use crate::utils::read_data::read_data;

pub fn init(dataset: &str) {
    println!("Starting PSO algorithm for dataset: {}", dataset);

    // Read dataset and initialize combinations
    let combinations = read_data(dataset).expect("Failed to read dataset");
    println!("Dataset read successfully. Number of combinations: {}", combinations.len());

    // Hyperparameters for swarm
    let swarm_size = 20;
    let particle_size = combinations[0].combination.len(); // Size of each particle is the size of the combination
    let w = 0.6; // Inertia weight
    let c1 = 0.2; // Cognitive weight
    let c2 = 0.2; // Social weight
    let epsilon = 0.1; // Epsilon for convergence
    let runs = 20; // Number of runs
    let epochs = 100; // Number of epochs
    let k = 5; // Number of neighbors to consider

    // Main loop for PSO
    println!("----------------------------------------");
    println!("Starting Global PSO algorithm...");
    let mut swarm = Swarm::new(swarm_size, particle_size, UpdateMode::Global);
    swarm.run_multiple(w, c1, c2, epsilon, &combinations, runs, epochs);
    println!("----------------------------------------");

    println!("Starting KNeighbor PSO algorithm...");
    let mut swarm = Swarm::new(swarm_size, particle_size, UpdateMode::KNeighbor(k));
    swarm.run_multiple(w, c1, c2, epsilon, &combinations, runs, epochs);
    println!("----------------------------------------");
}