use crate::algorithms::pso::structs::particle::Particle; // Import particle struct
pub use crate::structs::combination::Combination; // Import combination struct
pub use crate::algorithms::pso::structs::pso_mode::UpdateMode; // Import update mode enum

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

    fn find_k_nearest_best(&self, particle_index: usize, k: usize) -> Vec<bool> {
        let mut neighbors = self.particles.iter().enumerate()
            .filter(|&(i, _)| i != particle_index) // Exclude self
            .map(|(i, p)| (i, p.best_loss, &p.best_position))
            .collect::<Vec<_>>();
        
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // Sort by best loss (ascending)
        
        neighbors.into_iter().take(k).map(|(_, _, pos)| pos.clone()).next().unwrap_or_else(|| self.particles[particle_index].best_position.clone())
    }

    // Update the particles in the swarm
    pub fn update_particles(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>) {
        let k_neighbor_positions = if let UpdateMode::KNeighbor(k) = self.mode {
            Some(
                (0..self.particles.len())
                    .map(|i| self.find_k_nearest_best(i, k))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        for (i, particle) in self.particles.iter_mut().enumerate() {
            let best_position = match self.mode {
                UpdateMode::Global => &self.best_position,
                UpdateMode::KNeighbor(_) => &k_neighbor_positions.as_ref().unwrap()[i],
            };

            particle.update_pos(best_position, w, c1, c2);
            particle.update_loss(epsilon, combinations);
        }
    }

    // Take a step in the swarm
    pub fn step(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>) {
        self.update_particles(w, c1, c2, epsilon, combinations);
        self.update_global_best();
    }
}