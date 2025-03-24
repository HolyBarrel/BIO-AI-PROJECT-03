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

pub fn set_individual_loss(individual: &mut Combination, lookup_table: &Vec<Combination>) {
    // Calculate the loss for the individual based on the lookup table
    let mut loss = 0.0;
    for i in lookup_table.iter() {
        if i.combination == individual.combination {
            loss = i.loss;
            break;
        }
    }
    individual.loss = loss; 
}

pub fn get_fitness(individual: &Combination) -> (usize, f64) {
    let activated_columns = individual.combination.iter().filter(|&&b| b).count();
    (activated_columns, individual.loss)
}

pub fn init_population(gene_length: usize, population_size: usize) -> Vec<Combination> {
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; // Path to the CSV file
    let lookup_table = read_data::read_data(file_path).unwrap(); // Read data from the file and unwrap the Result
    let mut population: Vec<Combination> = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let combination = create_individual(gene_length, &lookup_table);
        population.push(combination);
    }
    population
}

pub fn bit_flip_mutation(individual: &mut Combination, mutation_probability: f64, lookup_table: &Vec<Combination>) {
    let mut rng = rand::rng();
    let gene_length = individual.combination.len();
    for i in 0..gene_length {
        if rng.random::<f64>() < mutation_probability {
            individual.combination[i] = !individual.combination[i];
        }
    }

    // Update the loss of the mutated individual
    set_individual_loss(individual, lookup_table);
}

pub fn moo_loop() {
    let mut terminate = false;
    let mut population = init_population(9, 50); // Initialize the population
    let mut generation = 0;
    let mut mutation_probability = 0.1; // Mutation probability

    // FastnondominatedSort of the initial population

    //Binary tournament selection of the population

    while terminate == false && generation < 1000 {
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

// Updates the crowding distance for a single objective.
// obj_index = 0 for the first objective (f1), 1 for the second objective (f2).
// It sorts the population by that objective, assigns infinite distance to boundary
// individuals, and updates interior individuals using the standard formula.
fn assign_crowding_distance_for_objective(
    obj_index: usize,
    normalized_fitnesses: &[(f64, f64)],
    crowding_distances: &mut [f64],
) {
    // Create a list of indices [0, 1, ..., population_size - 1].
    let mut sorted_indices: Vec<usize> = (0..crowding_distances.len()).collect();

    // Sort indices by the chosen objective (f1 or f2).
    sorted_indices.sort_by(|&i, &j| {
        let val_i = if obj_index == 0 {
            normalized_fitnesses[i].0
        } else {
            normalized_fitnesses[i].1
        };
        let val_j = if obj_index == 0 {
            normalized_fitnesses[j].0
        } else {
            normalized_fitnesses[j].1
        };
        val_i.partial_cmp(&val_j).unwrap()
    });

    // Assign infinite distance to the boundary solutions for this objective.
    crowding_distances[sorted_indices[0]] = f64::INFINITY;
    crowding_distances[sorted_indices[sorted_indices.len() - 1]] = f64::INFINITY;

    // For interior points, add the normalized difference of neighbors.
    // This implements:
    //   d(I_j) += [ f_m^(I_(j+1)) - f_m^(I_(j-1)) ] / [ f_m^max - f_m^min ]
    for k in 1..(sorted_indices.len() - 1) {
        let current_idx = sorted_indices[k];
        if crowding_distances[current_idx] == f64::INFINITY {
            continue;
        }

        // The previous and next individuals in the sorted list.
        let prev_idx = sorted_indices[k - 1];
        let next_idx = sorted_indices[k + 1];

        // Extract the normalized objective values for these neighbors.
        let (prev_f1, prev_f2) = normalized_fitnesses[prev_idx];
        let (next_f1, next_f2) = normalized_fitnesses[next_idx];


        // Compute the difference along the chosen objective.
        let difference = match obj_index {
            0 => next_f1 - prev_f1,  // f1 dimension
            1 => next_f2 - prev_f2,  // f2 dimension
            _ => unreachable!(),
        };

        // Accumulate the difference into the crowding distance for the current individual.
        crowding_distances[current_idx] += difference;
    }
}

/// Calculates the crowding distance for each individual in `last_front`.
/// The crowding distance is a measure of how isolated an individual is
/// from its neighbors in the objective space, ensuring the population
/// remains well-spread.
///
/// For each objective m, on interior individuals j, we use:
///   d(I_j) = d(I_j) + [ f_m^(I_(j+1)) - f_m^(I_(j-1)) ] / [ f_m^max - f_m^min ]
/// Boundary individuals for each objective get assigned an infinite distance.
pub fn crowding_distance(last_front: &Vec<Combination>) -> (Vec<Combination>, Vec<f64>) {
    let population_size = last_front.len();

    // 1) Gather each individual's (f1, f2) = (# of features, loss).
    let fitnesses: Vec<(usize, f64)> = last_front
        .iter()
        .map(|individual| get_fitness(individual))
        .collect();

    // 2) Compute min/max for each objective so we can normalize.
    let (min_f1, max_f1) = match (
        fitnesses.iter().map(|f| f.0).min(),
        fitnesses.iter().map(|f| f.0).max(),
    ) {
        (Some(min_val), Some(max_val)) => (min_val, max_val),
        _ => (0, 0),
    };

    let (min_f2, max_f2) = {
        let min_val = fitnesses
            .iter()
            .map(|f| f.1)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let max_val = fitnesses
            .iter()
            .map(|f| f.1)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        (min_val, max_val)
    };

    // 3) Normalize each individual’s fitness values so comparisons are fair.
    let normalized_fitnesses: Vec<(f64, f64)> = fitnesses
        .iter()
        .map(|&(f1, f2)| {
            let norm_f1 = if max_f1 != min_f1 {
                (f1 as f64 - min_f1 as f64) / (max_f1 - min_f1) as f64
            } else {
                0.0
            };
            let norm_f2 = if max_f2 != min_f2 {
                (f2 - min_f2) / (max_f2 - min_f2)
            } else {
                0.0
            };
            (norm_f1, norm_f2)
        })
        .collect();

    // 4) Prepare a vector to store crowding distances for each individual.
    let mut crowding_distances = vec![0.0; population_size];
    // 5) Update crowding distance for both objectives (f1 and f2).
    assign_crowding_distance_for_objective(0, &normalized_fitnesses, &mut crowding_distances);
    assign_crowding_distance_for_objective(1, &normalized_fitnesses, &mut crowding_distances);

    //Returns the last front and the crowding distances as a tuple
    return (last_front.clone(), crowding_distances.clone());
}

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

