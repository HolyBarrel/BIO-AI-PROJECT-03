pub use crate::structs::combination::Combination;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec<bool>, // Current position of the particle
    pub loss: f64, // Current loss of the particle
    pub velocity: Vec<f64>, // Velocity of the particle
    pub best_position: Vec<bool>, // Stored best position for the particle
    pub best_loss: f64, // Stored best loss for the particle
}

// Functions
impl Particle {
    // Create a new particle with random position and velocity
    pub fn new(size: usize) -> Self {
        let position = (0..size).map(|_| rand::random::<bool>()).collect::<Vec<bool>>();
        let velocity = (0..size).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        let loss = f64::MAX; // Initialize loss to a large value
        let best_position = position.clone(); // Initialize best position to current position
        let best_loss = loss; // Initialize best loss to current loss

        Particle {
            position,
            loss,
            velocity,
            best_position,
            best_loss,
        }
    }

    // Print the particle's information
    pub fn print(&self) {
        println!("Particle Position: {:?}", self.position);
        println!("Particle Loss: {:?}", self.loss);
        println!("Particle Velocity: {:?}", self.velocity);
        println!("Best Position: {:?}", self.best_position);
        println!("Best Loss: {:?}", self.best_loss);
    }
    
    /// Updates velocity of a Particle
    /// 
    /// global_best_position: Global best position
    /// w: Inertia weight
    /// c1: Cognitive weight
    /// c2: Social weight 
    pub fn update(&mut self, global_best_position: &Vec<bool>, w: f64, c1: f64, c2: f64) {
        
        for i in 0..self.velocity.len() {
            let r1 = rand::random::<f64>();
            let r2 = rand::random::<f64>();

            let cognitive_component = c1 * r1 * (self.best_position[i] as i32 as f64 - self.position[i] as i32 as f64);
            let social_component = c2 * r2 * (global_best_position[i] as i32 as f64 - self.position[i] as i32 as f64);
            self.velocity[i] = w * self.velocity[i] + cognitive_component + social_component;
        }

        // Update position
        for i in 0..self.position.len() {
            self.position[i] = self.position[i] ^ (self.velocity[i] > 0.0);
        }
    }

    /// Estimates fitness of a Particle
    /// 
    /// epsilon: represents the weight of the penalty for the number of features used
    /// combinations: lookup table for the possible combinations and their losses
    pub fn estimate_fitness(&mut self, epsilon: f64, combinations: Vec<Combination>) -> f64 {
        let x = self.position.iter().map(|&x| x as i32 as f64).collect::<Vec<f64>>();
        
        // Set T to the Loss of the combination matching current position
        let h_eT: f64 = combinations.iter().find(|&c| c.combination == self.position).unwrap().loss;

        // Set h_p to the number of columns used
        let h_p: f64 = x.iter().sum::<f64>();

        // Rerturn h_e * T(x) + epsilon * h_p(x)
        let fit: f64 = h_eT + epsilon * h_p;
        return fit
    }
}