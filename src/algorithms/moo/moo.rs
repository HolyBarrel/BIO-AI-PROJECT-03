use crate::structs::combination::{self, Combination}; // Import the Combination struct from the combination module
use crate::utils::read_data;
use rand::prelude::*;

/// The new struct embedding Combination-like fields plus rank/distance
#[derive(Debug, Clone)]
pub struct MooCombination {
    pub combination: Vec<bool>,
    pub loss: f64,
    pub rank: i8,
    pub crowding_distance: f64,
}

/// Conversion from Combination -> MooCombination
impl From<Combination> for MooCombination {
    fn from(c: Combination) -> MooCombination {
        MooCombination {
            combination: c.combination,
            loss: c.loss,
            rank: -1,
            crowding_distance: 0.0,
        }
    }
}

/// Creates a single MooCombination, either from an existing Combination in `data`
/// or as a brand new one with random bits.
pub fn create_individual(gene_length: usize, data: &Vec<Combination>) -> MooCombination {
    let mut activated_columns = vec![false; gene_length];
    let mut rng = rand::rng();

    // Randomly generate a bit vector of length = num_columns
    let num_columns = 9;
    for i in 0..num_columns {
        let random_value: f64 = rng.random_range(0.0..1.0);
        activated_columns[i] = random_value < 0.5;
    }

    if let Some(existing) = data.iter().find(|c| c.combination == activated_columns) {
        existing.clone().into()
    } else {
        // Otherwise create a brand new MooCombination
        MooCombination {
            combination: activated_columns,
            loss: 0.0,
            rank: -1,
            crowding_distance: 0.0,
        }
    }
}

/// Updates the `loss` in a MooCombination by looking it up in `lookup_table`.
pub fn set_individual_loss(individual: &mut MooCombination, lookup_table: &Vec<Combination>) {
    let mut loss = 0.0;
    for i in lookup_table.iter() {
        if i.combination == individual.combination {
            loss = i.loss;
            break;
        }
    }
    individual.loss = loss;
}

/// Helper function to get fitness (activated_columns, loss) from a MooCombination
pub fn get_fitness_moo(individual: &MooCombination) -> (usize, f64) {
    let activated_columns = individual.combination.iter().filter(|&&b| b).count();
    (activated_columns, individual.loss)
}

/// Initializes a population of MooCombination.
pub fn init_population(gene_length: usize, population_size: usize) -> Vec<MooCombination> {
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; 
    let lookup_table = read_data::read_data(file_path).unwrap();
    let mut population: Vec<MooCombination> = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let individual = create_individual(gene_length, &lookup_table);
        population.push(individual);
    }
    population
}

/// Bit-flip mutation
pub fn bit_flip_mutation(individual: &mut MooCombination, mutation_probability: f64, lookup_table: &Vec<Combination>) {
    let mut rng = rand::rng();
    let gene_length = individual.combination.len();

    for i in 0..gene_length {
        if rng.random::<f64>() < mutation_probability {
            individual.combination[i] = !individual.combination[i];
        }
    }
    // Update the loss of the mutated individual
    set_individual_loss(individual, lookup_table);

    // <TODO: update rank and crowding_distance outside this function if needed>
}

/// Example binary tournament selection that returns a single MooCombination.
/// This version assumes a list of fronts, each with a vector of MooCombination
/// and a parallel Vec<f64> for their distances. TODO store rank/distance
/// inside each MooCombination
pub fn binary_tournament_selection(fronts: &Vec<(Vec<MooCombination>, Vec<f64>)>) -> MooCombination {
    let mut rng = rand::rng();

    let front_1_index = rng.random_range(0..fronts.len());
    let front_2_index = rng.random_range(0..fronts.len());

    let front1 = &fronts[front_1_index].0;
    let front2 = &fronts[front_2_index].0;

    let crowding_distance1 = &fronts[front_1_index].1;
    let crowding_distance2 = &fronts[front_2_index].1;

    let parent_competitor1 = rng.random_range(0..front1.len());
    let parent_competitor2 = rng.random_range(0..front2.len());

    // Compare front indices (rank), then crowding distance if needed
    if front_1_index < front_2_index {
        front1[parent_competitor1].clone()
    } else if front_1_index == front_2_index {
        if crowding_distance1[parent_competitor1] > crowding_distance2[parent_competitor2] {
            front1[parent_competitor1].clone()
        } else {
            front2[parent_competitor2].clone()
        }
    } else {
        front2[parent_competitor2].clone()
    }
}

/// Main MOO loop (placeholder)
pub fn moo_loop() {
    let mut terminate = false;
    let mut population = init_population(9, 50); // Initialize the population
    let mut generation = 0;
    let mutation_probability = 0.1; // Example mutation probability

    // Example usage: mutate the first individual
    // bit_flip_mutation(&mut population[0], mutation_probability, &lookup_table);

    while !terminate && generation < 1000 {
        generation += 1;

        // TODO: 
        // 1) Generate offspring
        // 2) Combine parent + offspring
        // 3) FastNondominatedSort
        // 4) Elitism
        // 5) Crowding distance
        // 6) Next generation
    }
}

/// Example fast nondominated sort for MooCombination.
/// TODO: store rank in each MooCombination)
pub fn fast_nondominated_sort(population: &Vec<MooCombination>) -> Vec<Vec<MooCombination>> {
    let population_size = population.len();

    // 1) Precompute fitness for each individual
    let mut fitnesses: Vec<(usize, f64)> = Vec::with_capacity(population_size);
    for individual in population {
        fitnesses.push(get_fitness_moo(individual));
    }

    // 2) For each individual, store domination info
    let mut domination_counts = vec![0; population_size];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); population_size];

    // 3) Pairwise comparisons
    for i in 0..population_size {
        let (i_feature_count, i_loss) = fitnesses[i];
        for j in 0..population_size {
            if i == j { continue; }
            let (j_feature_count, j_loss) = fitnesses[j];

            // If i dominates j
            if (i_feature_count <= j_feature_count && i_loss <= j_loss) &&
               (i_feature_count < j_feature_count || i_loss < j_loss) {
                dominated_sets[i].push(j);
            }
            // Else if j dominates i
            else if (j_feature_count <= i_feature_count && j_loss <= i_loss) &&
                    (j_feature_count < i_feature_count || j_loss < i_loss) {
                domination_counts[i] += 1;
            }
        }
    }

    // 4) Identify the first front (domination_count == 0)
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front: Vec<usize> = Vec::new();
    for i in 0..population_size {
        if domination_counts[i] == 0 {
            first_front.push(i);
        }
    }
    fronts.push(first_front);

    // 5) Iteratively build subsequent fronts
    let mut current_front_idx = 0;
    while current_front_idx < fronts.len() && !fronts[current_front_idx].is_empty() {
        let mut next_front = Vec::new();
        for &p in &fronts[current_front_idx] {
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
        current_front_idx += 1;
    }

    // 6) Convert index fronts -> solution fronts
    let mut sorted_fronts: Vec<Vec<MooCombination>> = Vec::new();
    for front in fronts {
        let mut front_individuals = Vec::new();
        for idx in front {
            front_individuals.push(population[idx].clone());
        }
        sorted_fronts.push(front_individuals);
    }
    sorted_fronts
}



pub fn init() {
    // Initialize the multi-objective optimization algorithm
    let gene_length = 9;
    let population = init_population(gene_length, 50); // Initialize the population

    // Print the initialized population for debugging
    for individual in &population {
        println!("{:?}", individual);
    }
}