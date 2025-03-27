mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module
use crate::algorithms::soo::soo; // Import the hello function from soo module

fn main() {
    // Call the hello function from pso.rs
    soo::single_output_optimization(["breast_cancer_wisconsin_original", "titanic", "wine_quality_combined"]);
}