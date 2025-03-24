use crate::algorithms::pso::structs::particle::Particle; // Import particle struct

#[derive(Debug, Clone)]
pub struct Swarm {
    pub particles: Vec<Particle>, // Vector of particles in the swarm
    pub best_position: Vec<bool>, // Global best position found by the swarm
    pub best_loss: f64, // Global best loss found by the swarm
    pub size: usize, // Size of the swarm
}

// Functions
impl Swarm {
    // Create a new swarm with a given size and particle size
    pub fn new(size: usize, particle_size: usize) -> Self {
        let particles = (0..size).map(|_| Particle::new(particle_size)).collect::<Vec<Particle>>();
        let best_position = vec![false; particle_size]; // Initialize global best position
        let best_loss = f64::MAX; // Initialize global best loss to a large value

        Swarm {
            particles,
            best_position,
            best_loss,
            size,
        }
    }

    // Print the swarm's information
    pub fn print(&self) {
        println!("Swarm Size: {:?}", self.size);
        println!("Global Best Position: {:?}", self.best_position);
        println!("Global Best Loss: {:?}", self.best_loss);
    }
}