mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module
use crate::algorithms::moo::moo::init; // Import the hello function from moo module

fn main() {
    init();
    // Call the hello function from pso.rs
    //soo::single_output_optimization(["breast_cancer_wisconsin_original", "titanic", "wine_quality_combined"]);
    soo::multi_run_validation("breast_cancer_wisconsin_original");
}