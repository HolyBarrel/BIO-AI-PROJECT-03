pub use crate::structs::combination::Combination;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec<bool>, // Current position of the particle
    pub cost: f64, // Current cost of the particle
    pub velocity: Vec<f64>, // Velocity of the particle
    pub best_position: Vec<bool>, // Stored best position for the particle
    pub best_cost: f64, // Stored best cost for the particle
}

// Functions
impl Particle {
    // Create a new particle with random position and velocity
    pub fn new(size: usize) -> Self {
        let position = (0..size).map(|_| rand::random::<bool>()).collect::<Vec<bool>>();
        let velocity = (0..size).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        let cost = f64::MAX; // Initialize cost to a large value
        let best_position = position.clone(); // Initialize best position to current position
        let best_cost = cost; // Initialize best cost to current cost

        Particle {
            position,
            cost,
            velocity,
            best_position,
            best_cost,
        }
    }

    /// Updates velocity of a Particle
    /// 
    /// global_best_position: Global best position
    /// w: Inertia weight
    /// c1: Cognitive weight
    /// c2: Social weight 
    pub fn update_pos(&mut self, global_best_position: &[bool], w: f64, c1: f64, c2: f64) {
        for i in 0..self.velocity.len() {
            let r1 = rand::random::<f64>();
            let r2 = rand::random::<f64>();
            
            // We temporary change the vectors to floats to calculate the velocity
            let best_pos_val = if self.best_position[i] { 1.0 } else { 0.0 };
            let curr_pos_val = if self.position[i] { 1.0 } else { 0.0 };
            let global_best_val = if global_best_position[i] { 1.0 } else { 0.0 };
            
            // Vectors representing local and global memory
            let cognitive_component = c1 * r1 * (best_pos_val - curr_pos_val);
            let social_component = c2 * r2 * (global_best_val - curr_pos_val);

            self.velocity[i] = w * self.velocity[i] + cognitive_component + social_component;
        }

        // Update position based on probability (sigmoid-like behavior)
        for i in 0..self.position.len() {
            let probability = 1.0 / (1.0 + (-self.velocity[i]).exp()); // Sigmoid function
            self.position[i] = rand::random::<f64>() < probability;
        }
    }


    /// Estimates fitness of a Particle
    /// 
    /// epsilon: represents the weight of the penalty for the number of features used
    /// combinations: lookup table for the possible combinations and their costs
    pub fn update_cost(&mut self, epsilon: f64, combinations: &Vec<Combination>) {
        let x = self.position.iter().map(|&x| x as i32 as f64).collect::<Vec<f64>>();
        
        // Set T to the Cost of the combination matching current position
        let h_e_t: f64 = combinations
            .iter()
            .find(|&c| c.combination == self.position)
            .map(|c| c.loss)
            .unwrap_or(f64::MAX);


        // Set h_p to the number of columns used
        let h_p: f64 = x.iter().sum::<f64>();

        // Rerturn h_e * T(x) + epsilon * h_p(x)
        let fit: f64 = h_e_t + epsilon * h_p;
        
        self.cost = fit;
        if fit < self.best_cost {
            self.best_cost = fit;                       // Update the best cost of the candidate solution
            self.best_position = self.position.clone(); // Update the best position of the particle
        }
    }
}