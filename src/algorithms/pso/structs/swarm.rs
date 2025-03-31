use crate::algorithms::pso::structs::particle::Particle;
pub use crate::structs::combination::Combination;
pub use crate::algorithms::pso::structs::pso_mode::UpdateMode;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Swarm {
    pub particles: Vec<Particle>,   // Vector of particles in the swarm
    pub best_position: Vec<bool>,   // Global best position found by the swarm
    pub best_cost: f64,             // Global best cost found by the swarm
    gen_to_best: usize,             // Number of generations until the best solution is found
    pub size: usize,                // Size of the swarm (number of particles)
    pub mode: UpdateMode,           // Update mode (global or k-neighbor)
}

impl Swarm {
    pub fn new(size: usize, particle_size: usize, mode: UpdateMode) -> Self {
        let mut swarm = Swarm {
            particles: Vec::new(),  // Empty at first
            best_position: vec![false; particle_size],
            best_cost: f64::MAX, 
            size: size,
            gen_to_best: 0,
            mode,
        };

        swarm.initialize_particles(size, particle_size);

        swarm
    }
    
    fn initialize_particles(&mut self, size: usize, particle_size: usize) {
        self.particles = (0..size).map(|_| Particle::new(particle_size)).collect::<Vec<Particle>>();
        self.best_position = vec![false; particle_size]; // Reset the best position
        self.best_cost = f64::MAX;  // Reset the best cost
        self.gen_to_best = 0;       // Reset the generation count
    }

    pub fn update_global_best(&mut self, epoch: usize) {
        for particle in &self.particles {
            if particle.best_cost < self.best_cost {
                self.best_cost = particle.best_cost;
                self.best_position = particle.best_position.clone();
                self.gen_to_best = epoch;
            }
        }
    }
    

    fn find_k_nearest_best(&self, particle_index: usize, k: usize) -> Vec<bool> {
        let mut neighbors = self.particles.iter().enumerate()
            .filter(|&(i, _)| i != particle_index) 
            .map(|(i, p)| (i, p.best_cost, &p.best_position))
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
            particle.update_cost(epsilon, combinations);
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
                println!("Epoch {}: Global best cost: {}", epoch, self.best_cost);
                println!("Epoch {}: Global best position: {:?}", epoch, self.best_position);
            }
        }
        if print_result {
            println!("PSO algorithm completed. Final global best cost: {}", self.best_cost);
            println!("Number of generations until best solution: {}", self.gen_to_best);
            println!("Final global best position: {:?}", self.best_position);
        }
        return ;
    }

    pub fn run_multiple(&mut self, w: f64, c1: f64, c2: f64, epsilon: f64, combinations: &Vec<Combination>, runs: usize, epoch_per_run: usize) {
        let mut model_solutions: Vec<Vec<bool>> = Vec::new();
        let mut model_costs: Vec<f64> = Vec::new();
        let mut gen_to_solutions: Vec<usize> = Vec::new();
        let mut runtimes: Vec<f64> = Vec::new();
    
        for _run in 0..runs {
            self.initialize_particles(self.particles.len(), self.best_position.len());
    
            let start_time = Instant::now(); // Start timer
    
            self.perform_pso(w, c1, c2, epsilon, combinations, epoch_per_run, false, false);
    
            let elapsed_time = start_time.elapsed().as_secs_f64(); // Get elapsed time in seconds
            runtimes.push(elapsed_time); // Store runtime
    
            model_solutions.push(self.best_position.clone());
            model_costs.push(self.best_cost);
            gen_to_solutions.push(self.gen_to_best);
        }
    
        // Calculate Average no. of evaluations to solution
        let avg_gen_to_solution: f64 = gen_to_solutions
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>() / runs as f64;
        println!("AES: {:.6}", avg_gen_to_solution * self.size as f64);
    
        // Calculate Average Cost
        let avg_cost: f64 = model_costs.iter().sum::<f64>() / runs as f64;
        println!("MBF: {:.6}", 1.0 - avg_cost);
    
        // Calculate and print average runtime
        let avg_runtime: f64 = runtimes.iter().sum::<f64>() / runs as f64;
        println!("Average Runtime: {:.6} seconds", avg_runtime);
    
        // Best solution
        println!("----------------------------------------");
        let best_solution_index = model_costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        println!(
            "Best solution: {:?}",
            model_solutions[best_solution_index].iter().map(|&b| b as u8).collect::<Vec<u8>>()
        );
        println!("Best solution loss: {}", combinations
            .iter()
            .find(|comb| comb.combination == model_solutions[best_solution_index])
            .map(|comb| comb.loss)
            .unwrap_or(f64::NAN)
        );
        
        let best_solution_loss = model_costs[best_solution_index];
        println!("Best solution model loss: {}", best_solution_loss);
    
        // Mean solution with loss
        println!("----------------------------------------");
        let mean_solution_cost = model_costs.iter().copied().sum::<f64>() / model_costs.len() as f64;
        let mean_solution_index = model_costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mean_solution = model_solutions[mean_solution_index].clone();
        println!(
            "Mean solution: {:?}",
            mean_solution.iter().map(|&b| b as u8).collect::<Vec<u8>>()
        );
        println!("Mean solution loss: {}", mean_solution_cost);
    
        // Find the corresponding model_costs for the best solution and mean solution
        let mean_solution_cost_for_model = mean_solution_cost;
        println!("Mean solution cost: {}", mean_solution_cost_for_model);
    }    
}