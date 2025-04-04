mod utils;
mod structs;
mod algorithms;
use crate::algorithms::pso::pso; // Import the hello function from pso module
use crate::algorithms::moo::moo::init; // Import the hello function from moo module

fn main() {
// pso::init();
init();
}