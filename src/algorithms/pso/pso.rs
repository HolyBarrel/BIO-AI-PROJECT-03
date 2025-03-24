pub use crate::algorithms::pso::structs::particle::Particle;
pub use crate::algorithms::pso::structs::swarm::Swarm;

pub fn init() {
    println!("Starting PSO algorithm...");

    // Initialize the swarm
    let swarm_size = 20;
    let particle_size = 9;

    let mut swarm = Swarm::new(swarm_size, particle_size);
    swarm.print(); // Print the initial state of the swarm
}