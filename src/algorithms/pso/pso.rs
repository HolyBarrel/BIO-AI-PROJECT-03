// Import particle struct
pub use crate::algorithms::pso::structs::particle::Particle;

pub fn init() {
    println!("Starting PSO algorithm...");

    // Initialize and print a particle
    let particle = Particle::new(4);
    particle.print();
}