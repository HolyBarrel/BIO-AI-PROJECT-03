use crate::structs::combination::{self, Combination}; // Import the Combination struct from the combination module
use crate::utils::read_data;
use rand::prelude::*;
use plotters::prelude::*;

// Plotting library for visualizing the population
// Points are colored based on their rank.
pub fn plot_population(population: &[MooCombination], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Filter out individuals with infinite loss (empty feature sets).
    let valid: Vec<&MooCombination> = population.iter().filter(|ind| !ind.loss.is_infinite()).collect();
    if valid.is_empty() {
        println!("No valid individuals to plot (all have infinite loss).");
        return Ok(());
    }

    // Compute the range for the x-axis (number of active columns)
    let x_min = valid.iter().map(|ind| ind.combination.iter().filter(|&&b| b).count() as i32).min().unwrap();
    let x_max = valid.iter().map(|ind| ind.combination.iter().filter(|&&b| b).count() as i32).max().unwrap();

    // Compute the range for the y-axis (loss)
    let y_min = valid.iter().map(|ind| ind.loss).fold(f64::INFINITY, |a, b| a.min(b));
    let y_max = valid.iter().map(|ind| ind.loss).fold(f64::NEG_INFINITY, |a, b| a.max(b));

    // Create a drawing area with 800x600 pixels.
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("NSGA-II Population", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("Number of Active Columns")
        .y_desc("Loss")
        .draw()?;

    // Plot each valid individual as a circle.
    chart.draw_series(
        valid.iter().map(|ind| {
            // Use the raw count of active columns for x.
            let x = ind.combination.iter().filter(|&&b| b).count() as i32;
            let y = ind.loss;

            // Choose a color based on rank.
            let color = match ind.rank {
                0 => &RED,
                1 => &BLUE,
                2 => &GREEN,
                3 => &BLACK,
                _ => &MAGENTA,
            };

            Circle::new((x, y), 5, color.filled())
        })
    )?
    .label("Individuals")
    .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    // Optionally, add a legend.
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    // Save the file
    root.present()?;
    println!("Plot saved to {}", filename);

    Ok(())
}


#[derive(Debug, Clone)]
pub struct MooCombination {
    pub combination: Vec<bool>,
    pub loss: f64,
    pub rank: i8,
    pub crowding_distance: f64,
}

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
        // Otherwise create a new MooCombination
        MooCombination {
            combination: activated_columns,
            loss: 0.0,
            rank: -1,
            crowding_distance: 0.0,
        }
    }
}

pub fn single_point_crossover(parent1: &MooCombination, parent2: &MooCombination, data: &Vec<Combination>) -> MooCombination {
    let mut rng = rand::rng();
    let gene_length = parent1.combination.len();
    let crossover_point = rng.random_range(0..gene_length);

    let mut child_combination = Vec::with_capacity(gene_length);
    for i in 0..gene_length {
        if i < crossover_point {
            child_combination.push(parent1.combination[i]);
        } else {
            child_combination.push(parent2.combination[i]);
        }
    }


    if let Some(existing) = data.iter().find(|c| c.combination == child_combination) {
        existing.clone().into()
    } else {
        // Otherwise create a new MooCombination
        MooCombination {
            combination: child_combination,
            loss: 0.0,
            rank: -1,
            crowding_distance: 0.0,
        }
    }
}

/// Updates the `loss` in a MooCombination by looking it up in `lookup_table`.
pub fn set_individual_loss(individual: &mut MooCombination, lookup_table: &Vec<Combination>) {
    // Penalize the empty feature set.
    if individual.combination.iter().all(|&b| !b) {
        individual.loss = f64::INFINITY;
        return;
    }
    
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
pub fn get_fitness(individual: &MooCombination) -> (usize, f64) {
    let activated_columns = individual.combination.iter().filter(|&&b| b).count();
    if activated_columns == 0 {
        (usize::MAX, individual.loss)
    } else {
        (activated_columns, individual.loss)
    }
}


/// Initializes a population of MooCombination.
pub fn init_population(gene_length: usize, population_size: usize, lookup_table: &Vec<Combination>) -> Vec<MooCombination> {
    let mut population: Vec<MooCombination> = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let individual = create_individual(gene_length, &lookup_table);
        population.push(individual);
    }
    population
}

/// Bit-flip mutation
pub fn bit_flip_mutation(individual: &mut MooCombination, mutation_probability: f64) {
    let mut rng = rand::rng();
    let gene_length = individual.combination.len();

    for i in 0..gene_length {
        if rng.random::<f64>() < mutation_probability {
            individual.combination[i] = !individual.combination[i];
        }
    }
}

/// Binary tournament selection using rank and crowding_distance from MooCombination.
/// - Lower `rank` is better (i.e., front 0 is better than front 1).
/// - If both have the same `rank`, the one with higher `crowding_distance` is better.
///
/// # Arguments
/// * `population` - A slice of `MooCombination` where each individual
///                  has `rank` and `crowding_distance` already assigned.
pub fn binary_tournament_selection(population: &[MooCombination]) -> MooCombination {
    let mut rng = rand::rng();

    // 1) Pick two random indices from the population
    let i1 = rng.random_range(0..population.len());
    let i2 = rng.random_range(0..population.len());

    let ind1 = &population[i1];
    let ind2 = &population[i2];

    // 2) Compare their rank
    if ind1.rank < ind2.rank {
        ind1.clone()
    } else if ind2.rank < ind1.rank {
        ind2.clone()
    } else {
        // 3) If the rank is the same, compare crowding distance
        if ind1.crowding_distance > ind2.crowding_distance {
            ind1.clone()
        } else {
            ind2.clone()
        }
    }
}

/// Assigns crowding distance for one objective (`obj_index`) across all individuals.
///
/// - `obj_index = 0` -> uses the first objective (normalized_fitnesses[i].0).
/// - `obj_index = 1` -> uses the second objective (normalized_fitnesses[i].1).
/// - Boundary individuals get `crowding_distance = ∞` for that objective.
/// - Interior individuals add the difference between neighbors in sorted order.
fn assign_crowding_distance_for_objective(
    obj_index: usize,
    normalized_fitnesses: &[(f64, f64)],
    population: &mut [MooCombination],
) {
    // Create a list of indices [0, 1, ..., population.len() - 1].
    let mut sorted_indices: Vec<usize> = (0..population.len()).collect();

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

    // If there's fewer than 2 individuals, just return
    // (no interior points to compare).
    if population.len() < 2 {
        return;
    }

    // Assign infinite distance to boundary individuals for this objective.
    population[sorted_indices[0]].crowding_distance = f64::INFINITY;
    population[sorted_indices[sorted_indices.len() - 1]].crowding_distance = f64::INFINITY;

    // For interior points, add the normalized difference of neighbors.
    // This implements:
    //    d(I_j) += [ f_m^(I_(j+1)) - f_m^(I_(j-1)) ]
    for k in 1..(sorted_indices.len() - 1) {
        let current_idx = sorted_indices[k];
        // Skip if already ∞ from a previous objective
        if population[current_idx].crowding_distance == f64::INFINITY {
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
            0 => next_f1 - prev_f1, // f1 dimension
            1 => next_f2 - prev_f2, // f2 dimension
            _ => unreachable!(),
        };

        // Accumulate the difference into the crowding distance.
        population[current_idx].crowding_distance += difference;
    }
}

/// Sorts population by rank, then for each distinct rank,
/// calls `crowding_distance` on that sub-slice.
pub fn assign_crowding_distance_all(population: &mut [MooCombination]) { 
    // Sort population by rank so that rank=0 are first, then rank=1, etc.
    population.sort_by_key(|ind| ind.rank);

    // Iterate over contiguous segments (sub-slices) of the same rank.
    let mut i = 0;
    while i < population.len() {
        let current_rank = population[i].rank;
        // Advance j until the rank changes
        let mut j = i + 1;
        while j < population.len() && population[j].rank == current_rank {
            j += 1;
        }

        // Now population[i..j] are all individuals of rank = current_rank
        let front_slice = &mut population[i..j];
        // Compute crowding distance for just this front
        crowding_distance(front_slice);

        // Move to the next distinct rank
        i = j;
    }
}

/// Computes and stores the crowding distance in each `MooCombination` of `population`.
///
/// Steps:
///  1) Compute (f1, f2) for each individual.
///  2) Find min and max for each objective.
///  3) Normalize f1 and f2 for each individual.
///  4) Reset `crowding_distance` to 0 for each individual.
///  5) Call `assign_crowding_distance_for_objective` for both objectives.
///
/// Boundary individuals become ∞ for each objective. Interior individuals get
/// an accumulated difference.
pub fn crowding_distance(population_front: &mut [MooCombination]) {
    let pop_size = population_front.len();
    if pop_size == 0 {
        return;
    }

    // 1) Gather each individual's (f1, f2) = (# of features, loss).
    //    For example, f1 = number of selected columns, f2 = loss.
    let fitnesses: Vec<(usize, f64)> = population_front
        .iter()
        .map(|ind| get_fitness(ind))
        .collect();

    // 2) Compute min/max for each objective
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

    // 3) Normalize each individual's fitness values
    let normalized_fitnesses: Vec<(f64, f64)> = fitnesses
    .iter()
    .map(|&(f1, f2)| {
        let norm_f1 = if max_f1 != min_f1 {
            (f1 as f64 - min_f1 as f64) / (max_f1 - min_f1) as f64
        } else {
            0.0
        };
        // If loss is infinite, set the normalized loss to 1.0.
        let norm_f2 = if f2.is_infinite() {
            1.0
        } else if max_f2 != min_f2 {
            (f2 - min_f2) / (max_f2 - min_f2)
        } else {
            0.0
        };
        (norm_f1, norm_f2)
    })
    .collect();


    // 4) Reset crowding distance for each individual to 0.0
    for ind in population_front.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    // 5) Assign crowding distance for each objective
    assign_crowding_distance_for_objective(0, &normalized_fitnesses, population_front);
    assign_crowding_distance_for_objective(1, &normalized_fitnesses, population_front);
}

pub fn generate_offspring_population(population: &[MooCombination], mutation_probability: f64, data: &Vec<Combination>) -> Vec<MooCombination> {
    let mut offspring_population = Vec::new();
    let population_size = population.len();

    // Generate offspring using binary tournament selection and single-point crossover
    for _ in 0..population_size {
        let parent1 = binary_tournament_selection(population);
        let parent2 = binary_tournament_selection(population);
        // Single-point crossover
        let mut child = single_point_crossover(&parent1, &parent2, data);
        // Mutate the child individual with the given mutation probability
        let mut rng = rand::rng();
        if rng.random::<f64>() < mutation_probability {
            bit_flip_mutation(&mut child, mutation_probability);
        }
        set_individual_loss(&mut child, data); 
        offspring_population.push(child);
    }

    offspring_population
}

/// Performs fast nondominated sorting on `population` and assigns each individual's `rank`
/// according to the front it belongs to. Returns a flattened Vec<MooCombination> where
/// `rank = 0` means the first (best) front, `rank = 1` means the second front, etc.
pub fn fast_nondominated_sort(mut population: Vec<MooCombination>) -> Vec<MooCombination> {
    let population_size = population.len();

    // 1) Precompute fitness (f1, f2) for each individual
    //    (here: f1 = number of features, f2 = loss).
    let mut fitnesses: Vec<(usize, f64)> = Vec::with_capacity(population_size);
    for individual in &population {
        fitnesses.push(get_fitness(individual));
    }

    // 2) For each individual, store domination info
    let mut domination_counts = vec![0; population_size];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); population_size];

    // 3) Pairwise comparisons
    for i in 0..population_size {
        let (i_feature_count, i_loss) = fitnesses[i];
        for j in 0..population_size {
            if i == j {
                continue;
            }
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

    // 6) Assign ranks and flatten
    //    front 0 -> rank = 0, front 1 -> rank = 1, etc.
    let mut rank_value = 0;
    for front in &fronts {
        for &idx in front {
            population[idx].rank = rank_value;
        }
        rank_value += 1;
    }

    // 7) Return the updated population in a single Vec<MooCombination>
    population
}

pub fn merge_populations(parent_population: &[MooCombination], offspring_population: &[MooCombination]) -> Vec<MooCombination> {
    let mut merged_population = Vec::new();
    merged_population.extend_from_slice(parent_population);
    merged_population.extend_from_slice(offspring_population);
    merged_population
}

/// Returns the fronts as a Vec of Vecs, where each inner Vec contains individuals of the same rank.
pub fn fast_nondominated_sort_by_front(population: &[MooCombination]) -> Vec<Vec<MooCombination>> {
    let mut pop = population.to_vec();
    pop = fast_nondominated_sort(pop); // This sets the ranks appropriately.
    
    // Group individuals by their rank.
    pop.sort_by_key(|ind| ind.rank);
    let mut fronts: Vec<Vec<MooCombination>> = Vec::new();
    let mut current_front: Vec<MooCombination> = Vec::new();
    let mut current_rank = pop[0].rank;
    
    for ind in pop {
        if ind.rank != current_rank {
            fronts.push(current_front);
            current_front = Vec::new();
            current_rank = ind.rank;
        }
        current_front.push(ind);
    }
    if !current_front.is_empty() {
        fronts.push(current_front);
    }
    
    fronts
}

/// Performs elitism by adding full fronts until the desired size is reached.
/// For the last front, computes crowding distances and selects the best individuals.
pub fn elitism(super_population: &[MooCombination]) -> Vec<MooCombination> {
    let next_population_size = super_population.len() / 2;
    let fronts = fast_nondominated_sort_by_front(super_population);
    let mut new_population = Vec::with_capacity(next_population_size);

    for front in fronts {
        if new_population.len() + front.len() <= next_population_size {
            // Add the entire front.
            new_population.extend(front);
        } else {
            let mut last_front = front.clone();
            crowding_distance(&mut last_front);
            // Sort by descending crowding distance.
            last_front.sort_by(|a, b| b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap());
            let remaining = next_population_size - new_population.len();
            new_population.extend(last_front.into_iter().take(remaining));
            break;
        }
    }
    new_population
}

pub fn init() {
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; 
    let lookup_table = read_data::read_data(file_path).unwrap();
    let mut terminate = false;
    let mut population = init_population(9, 500, &lookup_table); // Initialize the population

    //TODO; set rank and crowding distance for each individual in the population P_0

    // Call the fast_nondominated_sort function to sort the population
    let mut sorted_population = fast_nondominated_sort(population.clone());

    // Assgns the crowding distance for the sorted population
    assign_crowding_distance_all(&mut sorted_population);

    let mut generation = 0;
    let mutation_probability = 0.1;

    // Example usage: mutate the first individual
    // bit_flip_mutation(&mut population[0], mutation_probability, &lookup_table);

    population = sorted_population.clone();

    //TODO: fix possibility for individuals to get 0,0


    while !terminate && generation < 1 {
        generation += 1;
        // Prints for every 1 generations
        if generation % 100 == 0 {
            println!("Generation: {}", generation);
            // for individual in &population {
            //    //all the attrs of the individual
            //     println!("Combination: {:?}, Loss: {}, Rank: {}, Crowding Distance: {}", individual.combination, individual.loss, individual.rank, individual.crowding_distance);
            // }
 
        }

        // 1 ) Generate offspring population
        let offspring_population = generate_offspring_population(&population, mutation_probability, &lookup_table);

        // 2) Merge parent and offspring populations
        let super_population = merge_populations(&population, &offspring_population);

        // 3) Perform fast nondominated sorting on the merged population
        let sorted_super_population = fast_nondominated_sort(super_population.clone());

        // 4) Apply elitism to select the best individuals for the next generation
        let new_population = elitism(&sorted_super_population);

        // 5 assigns the population to the new population
        population = new_population.clone();

    }

    // 6) Perform fast nondominated sorting on the final population
    let mut final_sorted_population = fast_nondominated_sort(population.clone());
    // 7) Assign crowding distance to the final population
    assign_crowding_distance_all(&mut final_sorted_population);
    // Prints the final population
    println!("Final Population:");

    for individual in &final_sorted_population {
        println!("Combination: {:?}, Loss: {}, Rank: {}, Crowding Distance: {}", individual.combination, individual.loss, individual.rank, individual.crowding_distance);
    }

    // 8) Plot the final population
    let plot_filename = "XGB-Feature-Selection/output/test.png";
    plot_population(&final_sorted_population, plot_filename).unwrap();


}