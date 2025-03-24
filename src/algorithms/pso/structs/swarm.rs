use crate::algorithms::pso::structs::particle::Particle; // Import particle struct
pub use crate::structs::combination::Combination; // Import combination struct

#[derive(Debug, Clone, Copy)]
pub enum UpdateMode {
    Global,
    KNeighbor(usize),
}

#[derive(Debug, Clone)]
pub struct Swarm {
    pub particles: Vec<Particle>, // Vector of particles in the swarm
    pub best_position: Vec<bool>, // Global best position found by the swarm
    pub best_loss: f64, // Global best loss found by the swarm
    pub size: usize, // Size of the swarm
    pub mode: UpdateMode, // Update mode (global or k-neighbor)
}

impl Swarm {
    pub fn new(size: usize, particle_size: usize, mode: UpdateMode) -> Self {
        let particles = (0..size).map(|_| Particle::new(particle_size)).collect::<Vec<Particle>>();
        let best_position = vec![false; particle_size]; // Initialize global best position
        let best_loss = f64::MAX; // Initialize global best loss to a large value

        Swarm {
            particles,
            best_position,
            best_loss,
            size,
            mode,
        }
    }

    pub fn print(&self) {
        println!("Swarm Size: {:?}", self.size);
        println!("Global Best Position: {:?}", self.best_position);
        println!("Global Best Loss: {:?}", self.best_loss);
    }

    pub fn update_global_best(&mut self) {
        for particle in &self.particles {
            if particle.best_loss < self.best_loss {
                self.best_loss = particle.best_loss;
                self.best_position = particle.best_position.clone();
            }
        }
    }

    // Update the particles in the swarm
    pub fn update_particles(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>) {
        for particle in &mut self.particles {
            particle.update_pos(&self.best_position, w, c1, c2);
            // Update the particle's loss based on its position
            particle.update_loss(epsilon, combinations);
        }
    }

    // Take a step in the swarm
    pub fn step(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>) {
        self.update_particles(w, c1, c2, epsilon, combinations);
        self.update_global_best();
    }
}