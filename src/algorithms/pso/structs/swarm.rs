use crate::algorithms::pso::structs::particle::Particle;
pub use crate::structs::combination::Combination;
pub use crate::algorithms::pso::structs::pso_mode::UpdateMode;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Swarm {
    pub particles: Vec<Particle>,   // Vector of particles in the swarm
    pub best_position: Vec<bool>,   // Global best position found by the swarm
    pub best_loss: f64,             // Global best loss found by the swarm
    gen_to_best: usize,             // Number of generations until the best solution is found
    pub size: usize,                // Size of the swarm
    pub mode: UpdateMode,           // Update mode (global or k-neighbor)
}

impl Swarm {
    pub fn new(size: usize, particle_size: usize, mode: UpdateMode) -> Self {
        let particles = (0..size).map(|_| Particle::new(particle_size)).collect::<Vec<Particle>>();
        let best_position = vec![false; particle_size];
        let best_loss: f64 = f64::MAX; 
        let gen_to_best: usize = 0;

        Swarm {
            particles,
            best_position,
            best_loss,
            size,
            gen_to_best,
            mode,
        }
    }

    pub fn print(&self) {
        println!("Swarm Size: {:?}", self.size);
        println!("Global Best Position: {:?}", self.best_position);
        println!("Global Best Loss: {:?}", self.best_loss);
    }

    pub fn update_global_best(&mut self, epoch: usize) {
        for particle in &self.particles {
            if particle.best_loss < self.best_loss {
                self.best_loss = particle.best_loss;
                self.best_position = particle.best_position.clone();
                self.gen_to_best = epoch;
            }
        }
    }

    fn find_k_nearest_best(&self, particle_index: usize, k: usize) -> Vec<bool> {
        let mut neighbors = self.particles.iter().enumerate()
            .filter(|&(i, _)| i != particle_index) 
            .map(|(i, p)| (i, p.best_loss, &p.best_position))
            .collect::<Vec<_>>();
        
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
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
    pub fn step(&mut self, epoch: usize, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>) {
        self.update_particles(w, c1, c2, epsilon, combinations);
        self.update_global_best(epoch);
    }

    pub fn perform_pso(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>, epochs: usize, print_performance: bool, print_result: bool) {
        for epoch in 0..epochs {
            self.step(epoch, w, c1, c2, epsilon, combinations);
            if epoch % 100 == 0 && print_performance {
                println!("Epoch {}: Global best loss: {}", epoch, self.best_loss);
                println!("Epoch {}: Global best position: {:?}", epoch, self.best_position);
            }
        }
        if print_result {
            println!("PSO algorithm completed. Final global best loss: {}", self.best_loss);
            println!("Number of generations until best solution: {}", self.gen_to_best);
            println!("Final global best position: {:?}", self.best_position);
        }
        return ;
    }

    pub fn run_multiple(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>, runs: usize, epoch_per_run: usize) {
        let mut model_solutions: Vec<Vec<bool>> = Vec::new();
        let mut model_losses: Vec<f64> = Vec::new();
        let mut gen_to_solutions: Vec<usize> = Vec::new();
        let mut runtimes: Vec<f64> = Vec::new(); // Store runtime for each run
    
        for run in 0..runs {
            // Reset the swarm's state before starting each run
            self.best_position = vec![false; self.best_position.len()];
            self.best_loss = f64::MAX;
            self.gen_to_best = 0;

            let start_time = Instant::now(); // Start timer
    
            self.perform_pso(w, c1, c2, epsilon, combinations, epoch_per_run, false, false);
            
            let elapsed_time = start_time.elapsed().as_secs_f64(); // Get elapsed time in seconds
            runtimes.push(elapsed_time); // Store runtime
    
            model_solutions.push(self.best_position.clone());
            model_losses.push(self.best_loss);
            gen_to_solutions.push(self.gen_to_best);
        }
    
        // Calculate Average no. of evaluations to solution
        let avg_gen_to_solution: f64 = gen_to_solutions.iter().map(|&x| x as f64).sum::<f64>() / runs as f64;
        println!("AES: {:.6}", avg_gen_to_solution);
    
        // Calculate Average loss
        let avg_loss: f64 = model_losses.iter().sum::<f64>() / runs as f64;
        println!("MBF: {:.6}", 1.0 - avg_loss);
    
        // Calculate and print average runtime
        let avg_runtime: f64 = runtimes.iter().sum::<f64>() / runs as f64;
        println!("Average Runtime: {:.6} seconds", avg_runtime);
    }
}