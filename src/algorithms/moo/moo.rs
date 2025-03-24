use crate::structs::combination::{self, Combination}; // Import the Combination struct from the combination module
use crate::utils::read_data;
use rand::prelude::*;

pub fn create_individual(gene_length: usize, data: &Vec<Combination>) -> Combination {
    let mut activated_columns = vec![false; gene_length];
    let mut rng = rand::rng();

    let num_columns = 9;
    for i in 0..num_columns {
        let random_value: f64 = rng.random_range(0.0..1.0);
        if random_value < 0.5 {
            activated_columns[i] = true; 
        } else {
            activated_columns[i] = false; 
        }
    } 

    for i in data.iter() {
        if i.combination == activated_columns {
            return i.clone();
        }
    }

    // If no matching combination is found, return a new individual with the activated columns
    Combination {
        combination: activated_columns,
        loss: 0.0, // Initialize loss to 0.0 if none is found
    }

}

pub fn get_fitness(individual: &Combination) -> (usize, f64) {
    let activated_columns = individual.combination.iter().filter(|&&b| b).count();
    (activated_columns, individual.loss)
}

pub fn init_population(gene_length: usize, population_size: usize) -> Vec<Combination> {
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; // Path to the CSV file
    let data = read_data::read_data(file_path).unwrap(); // Read data from the file and unwrap the Result
    let mut population: Vec<Combination> = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let combination = create_individual(gene_length, &data);
        population.push(combination);
    }
    population
}

pub fn moo_loop() {
    let mut terminate = false;
    let mut population = init_population(9, 50); // Initialize the population
    let mut generation = 0;
    while terminate == false && generation < 100 {
        generation += 1;

        // Generate offspring

        // Combine parent and offspring populations to superpopulation S with size 2 * population_size (200)

        // FastNondominatedSort of S

        // Create new population P from the first n individuals using elitism

        // For the last front, the individuals are selected using the crowding distance

        // the new population P is the new population for the next generation
    }
}

pub fn fast_nondominated_sort(population: &Vec<Combination>) -> Vec<Vec<Combination>> {
    // Precompute fitness values for all individuals:
    // Each fitness is a tuple: (number of activated columns, loss)
    let mut fitnesses: Vec<(usize, f64)> = Vec::with_capacity(population.len());
    for individual in population {
        fitnesses.push(get_fitness(individual));
    }

    let population_size = population.len();
    // For each individual, store:
    // domination_count[i]: number of individuals dominating i
    // dominated_sets[i]: indices of individuals that i dominates
    let mut domination_counts = vec![0; population_size];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); population_size];

    // Pairwise comparisons: use the precomputed fitnesses.
    for i in 0..population_size {
        let (i_feature_count, i_loss) = fitnesses[i];
        for j in 0..population_size {
            if i == j {
                continue; // Skip self-comparison
            }
            let (j_feature_count, j_loss) = fitnesses[j];
            // Check if individual i dominates individual j
            if (i_feature_count <= j_feature_count && i_loss <= j_loss) &&
               (i_feature_count < j_feature_count || i_loss < j_loss) {
                dominated_sets[i].push(j);
            } 
            // Else, check if j dominates i
            else if (j_feature_count <= i_feature_count && j_loss <= i_loss) &&
                    (j_feature_count < i_feature_count || j_loss < i_loss) {
                domination_counts[i] += 1;
            }
        }
    }

    // Identify the first front (non-dominated individuals): indices with domination count zero.
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front: Vec<usize> = Vec::new();
    for i in 0..population_size {
        if domination_counts[i] == 0 {
            first_front.push(i);
        }
    }
    fronts.push(first_front);

    // Iteratively build subsequent fronts.
    let mut i = 0;
    while i < fronts.len() && !fronts[i].is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_sets[p] {
                domination_counts[q] -= 1;
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        i += 1;
    }

    // Convert fronts (indices) to actual individuals.
    let mut sorted_fronts: Vec<Vec<Combination>> = Vec::new();
    for front in fronts {
        let mut front_individuals: Vec<Combination> = Vec::new();
        for &idx in &front {
            front_individuals.push(population[idx].clone());
        }
        sorted_fronts.push(front_individuals);
    }

    sorted_fronts


}

// pub fn crowding_distance(population: &Vec<Combination>) -> Vec<Combination> {

// }


pub fn init() {
    // Initialize the multi-objective optimization algorithm
    let gene_length = 9;
    let population = init_population(gene_length, 50); // Initialize the population

    // // Print the initialized population for debugging
    // for individual in &population {
    //     println!("{:?}", individual);
    //     // Prints the individual get fitness
    //     let (activated_columns, loss) = get_fitness(individual);
    //     println!("Activated columns: {}, Loss: {}", activated_columns, loss);
    // }

    // Runs the fast nondominated sort on the population
    let sorted_fronts = fast_nondominated_sort(&population);

    // Print the sorted fronts for debugging
    for (i, front) in sorted_fronts.iter().enumerate() {
        println!("Front {}: {:?}", i, front);
        // Print the fitness of each individual in the front
        for individual in front {
            // Prints the individual get fitness
            // Prints front index
            print!("Front {}: ", i);
            let (activated_columns, loss) = get_fitness(individual);
            println!("Activated columns: {}, Loss: {}", activated_columns, loss);
        }
    }
}

