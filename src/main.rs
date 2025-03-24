mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module

fn main() {
    // Call the hello function from pso.rs
    pso::init();
}
