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
}