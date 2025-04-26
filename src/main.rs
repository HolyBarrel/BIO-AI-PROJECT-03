mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module
use crate::algorithms::moo::moo::init; // Import the hello function from moo module
use crate::algorithms::soo::soo;

fn main() {
    //init();
    // Call the hello function from pso.rs
    //soo::single_output_optimization(["breast_cancer_wisconsin_original", "titanic", "wine_quality_combined"]);
    //soo::single_output_test_optimization("random_forest_zoo.csv");
    init();

}