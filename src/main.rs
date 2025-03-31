mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module
use crate::algorithms::moo::moo::init; // Import the hello function from moo module
use crate::algorithms::soo::soo;

fn main() {
    //init();
    // Call the hello function from pso.rs
    pso::init("XGB-Feature-Selection/output/wine_quality_combined");
}