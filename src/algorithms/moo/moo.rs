use crate::structs::combination::Combination; // Import the Combination struct from the combination module
use crate::utils::read_data;
use rand::prelude::*;
use plotters::prelude::*;
use std::collections::HashMap;
use std::time::{Instant, Duration};
use serde::Serialize; // Import Serialize derive macro


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
    let mut rng = rand::rng();

    let mut activated_columns = vec![false; gene_length];

    let num_active = rng.random_range(1..=gene_length);

    // Create a list of indices and shuffle it.
    let mut indices: Vec<usize> = (0..gene_length).collect();
    indices.shuffle(&mut rng);

    // Set the first num_active indices to true.
    for &i in indices.iter().take(num_active) {
        activated_columns[i] = true;
    }


    if let Some(existing) = data.iter().find(|c| c.combination == activated_columns) {
        existing.clone().into()
    } else {
        println!("Creating new individual: {:?}", activated_columns);
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

    // Ensure at least one feature is selected after mutation
    enforce_valid_combination(individual);
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

pub fn enforce_valid_combination(individual: &mut MooCombination) {
    // Ensure that at least one feature is selected.
    if individual.combination.iter().all(|&b| !b) {
        let random_index = rand::rng().random_range(0..individual.combination.len());
        individual.combination[random_index] = true;
    }
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

/**
 * Plots the population of individuals for the MOO algorithm.
 * Each individual is represented as a point in a 2D space, where the x-axis represents the number of active features (columns) 
 * and the y-axis represents the loss value.
 * The points are colored based on their rank in the population.
 */
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
        .caption("NSGA-II Population", ("sans-serif", 50))
        .margin(20)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max + 1, y_min..y_max + 0.01)?;

        chart.configure_mesh()
        .x_desc("Number of Active Features (Columns)")
        .y_desc("Loss")
        
        // Set the font size for the axis labels
        .axis_desc_style(("sans-serif", 22).into_font())
        
        // Increase tick label font size
        .x_label_style(("sans-serif", 20).into_font())
        .y_label_style(("sans-serif", 20).into_font())
        .draw()?;

    // Plot each valid individual as a circle.
    chart.draw_series(
        valid.iter().map(|ind| {
            // Compute x and y values
            let x = ind.combination.iter().filter(|&&b| b).count() as i32;
            let y = ind.loss;
            // Choose a color based on rank.
            let color = match ind.rank {
                0 => &RED,
                1 => &MAGENTA,
                2 => &BLUE,
                3 => &GREEN,
                _ => &BLACK,
            };
            EmptyElement::at((x, y))
                + Circle::new((0, 0), 5, color.filled())
                + Circle::new((0, 0), 5, ShapeStyle::from(&BLACK).stroke_width(1))
        })
    )?
    .label("Individuals")
    // // Add a legend to the plot and the size/color of the dot
    .legend(|(x, y)| Circle::new((x, y), 5, ShapeStyle::from(&BLACK).stroke_width(1)));

    // Configures the legend
    chart.configure_series_labels()
    .border_style(&BLACK)
    .label_font(("sans-serif", 20).into_font())
    .draw()?;

    root.present()?;
    println!("Plot saved to {}", filename);

    Ok(())
}

/// Generic MOO algorithm runner.
/// It initializes the population, then iteratively evolves it until the provided termination
/// condition (a closure that takes the current generation count and the start time) returns false.
/// The closure can use the current generation and elapsed time to decide when to stop.
fn run_moo_algorithm_generic<F>(
    file_path: &str,
    population_size: usize,
    gene_length: usize,
    generations_to_print: usize,
    print: bool,
    mut termination_condition: F,
) -> Vec<MooCombination>
where
    F: FnMut(usize, Instant) -> bool,
{
    // Load dataset and initialize population.
    let lookup_table = read_data::read_data(file_path).unwrap();
    let mut population = init_population(gene_length, population_size, &lookup_table);

    // Initial nondominated sort and assign crowding distances.
    let mut sorted_population = fast_nondominated_sort(population.clone());
    assign_crowding_distance_all(&mut sorted_population);
    let mutation_probability = 0.1;
    population = sorted_population.clone();

    // Record start time for time-based termination.
    let start_time = Instant::now();
    let mut generation = 0;

    // Run evolutionary loop until termination condition returns false.
    while termination_condition(generation, start_time) {
        generation += 1;

        // Print generation info if needed.
        if generation % generations_to_print == 0 && print {
            println!("Generation: {}", generation);
        }

        // 1) Generate offspring.
        let offspring_population = generate_offspring_population(&population, mutation_probability, &lookup_table);
        // 2) Merge parent and offspring populations.
        let super_population = merge_populations(&population, &offspring_population);
        // 3) Sort the merged population.
        let sorted_super_population = fast_nondominated_sort(super_population.clone());
        // 4) Apply elitism to select the next generation.
        let new_population = elitism(&sorted_super_population);
        population = new_population.clone();
    }

    // Final nondominated sorting and crowding distance assignment.
    let mut final_sorted_population = fast_nondominated_sort(population.clone());
    assign_crowding_distance_all(&mut final_sorted_population);

    if print {
        println!("Final Population:");
        for individual in &final_sorted_population {
            println!(
                "Combination: {:?}, Loss: {}, Rank: {}, Crowding Distance: {}",
                individual.combination, individual.loss, individual.rank, individual.crowding_distance
            );
        }
    }
    final_sorted_population
}

/// Runs the MOO algorithm for a fixed number of generations.
pub fn run_moo_algorithm(
    file_path: &str,
    population_size: usize,
    generations: usize,
    generations_to_print: usize,
    gene_length: usize,
    print: bool,
) -> Vec<MooCombination> {
    run_moo_algorithm_generic(
        file_path,
        population_size,
        gene_length,
        generations_to_print,
        print,
        |generation, _start_time| generation < generations,
    )
}

/// Runs the MOO algorithm for a fixed time limit (in seconds).
pub fn run_moo_algorithm_time(
    file_path: &str,
    population_size: usize,
    time_limit: u64, // time limit in seconds
    generations_to_print: usize,
    gene_length: usize,
    print: bool,
) -> Vec<MooCombination> {
    run_moo_algorithm_generic(
        file_path,
        population_size,
        gene_length,
        generations_to_print,
        print,
        |_, start_time| start_time.elapsed() < Duration::from_secs(time_limit),
    )
}

pub fn execute_run_n_times(
    n_times: usize, 
    population_size: usize,
    generations: usize, 
    generations_to_print: usize, 
    gene_length: usize, 
    file_path: &str,
    print: bool,
    timed: bool
) -> Vec<MooCombination> {
    let mut all_best: Vec<MooCombination> = Vec::new();

    for _ in 0..n_times {
        let mut best_from_run: Vec<MooCombination> = Vec::new();
        if timed {
            best_from_run = run_moo_algorithm_time(file_path, population_size, 1, generations_to_print, gene_length, print);
        }
        else {
            best_from_run = run_moo_algorithm(file_path, population_size, generations, generations_to_print, gene_length, print);
        }
        all_best.extend(best_from_run);
    }
    all_best
}

#[derive(Serialize)]
struct RunMetric {
    loss: f64,
    num_features: usize,
    frequency: usize,
    percentage: f64,
    combination: String,
}

pub fn save_run_metrics_to_csv(
    all_best: &[MooCombination],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build a frequency map for unique individuals (using a key based on number of active features and loss).
    let mut freq_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for ind in all_best {
        if ind.loss.is_infinite() {
            continue;
        }
        let (active, loss) = get_fitness(ind);
        // Key format: "(num_features, loss)" with loss rounded to 3 decimal places.
        let key = format!("({}, {:.3})", active, loss);
        *freq_map.entry(key).or_insert(0) += 1;
    }
    // Total count of valid individuals (for percentage calculations).
    let total: usize = freq_map.values().copied().sum();

    // Create a map for unique individuals.
    let mut unique_points: std::collections::HashMap<String, &MooCombination> = std::collections::HashMap::new();
    for ind in all_best {
        if ind.loss.is_infinite() {
            continue;
        }
        let (active, loss) = get_fitness(ind);
        let key = format!("({}, {:.3})", active, loss);
        unique_points.entry(key).or_insert(ind);
    }

    let mut wtr = csv::Writer::from_path(filename)?;
    // For each unique solution, write a row with its loss, num_features, frequency, percentage, and gene combination.
    for (key, ind) in unique_points {
        let frequency = *freq_map.get(&key).unwrap_or(&0);
        let percentage = ((frequency as f64 / total as f64) * 100.0 * 100.0).round() / 100.0;
        let (active, loss) = get_fitness(ind);
        // Convert the gene combination (vector of booleans) into a comma-separated string.
        let combination_str = ind
            .combination
            .iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<Vec<&str>>()
            .join(",");
        let metric = RunMetric {
            loss,
            num_features: active,
            frequency,
            percentage,
            combination: combination_str,
        };
        wtr.serialize(metric)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn plot_best_pareto_overview(
    all_best: &[MooCombination],
    filename: &str,
    plot_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build frequency map from the all_best population.
    let mut freq_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for ind in all_best {
        // Skip individuals with infinite loss.
        if ind.loss.is_infinite() {
            continue;
        }
        let (active, loss) = get_fitness(ind);
        // Create a key that identifies a point (using a rounded loss to 3 decimal places).
        let key = format!("({}, {:.3})", active, loss);
        *freq_map.entry(key).or_insert(0) += 1;
    }
    println!("{:?}", freq_map);

    // Extract unique points from the all_best population.
    let mut unique_points: std::collections::HashMap<String, &MooCombination> = std::collections::HashMap::new();
    for ind in all_best {
        if ind.loss.is_infinite() {
            continue;
        }
        let (active, loss) = get_fitness(ind);
        let key = format!("({}, {:.3})", active, loss);
        unique_points.entry(key).or_insert(ind);
    }
    
    let unique_vals: Vec<&&MooCombination> = unique_points.values().collect();
    if unique_vals.is_empty() {
        println!("No valid individuals to plot.");
        return Ok(());
    }
    
    // Determine plot ranges.
    let x_min = unique_vals
        .iter()
        .map(|ind| ind.combination.iter().filter(|&&b| b).count() as i32)
        .min()
        .unwrap();
    let x_max = unique_vals
        .iter()
        .map(|ind| ind.combination.iter().filter(|&&b| b).count() as i32)
        .max()
        .unwrap();
    let y_min = unique_vals
        .iter()
        .map(|ind| ind.loss)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let y_max = unique_vals
        .iter()
        .map(|ind| ind.loss)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    
    // Find maximum frequency for grouping.
    let max_freq = freq_map.values().copied().max().unwrap_or(1);
    
    // Set up discrete groups dynamically.
    let tick_count = 5;
    let discrete_colors = vec![
        RGBColor(0, 0, 0),    // Black
        RGBColor(0, 255, 0),  // Green
        RGBColor(0, 0, 255),  // Blue
        RGBColor(255, 0, 255),    // Magenta
        RGBColor(255, 0, 0),    // Red
    ];
    let group_interval = max_freq as f64 / tick_count as f64;
    
    // Set up the drawing area and split it for the chart and legend.
    let root = BitMapBackend::new(filename, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_horizontally(750); 
    let chart_area = areas.0.clone();
    let legend_area = areas.1.clone();
    
    let plot_title = format!("{} Pareto Front Overview", plot_name);
    let mut chart = ChartBuilder::on(&chart_area)
        .caption(plot_title, ("sans-serif", 50))
        .margin(20)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..(x_max + 1), y_min..(y_max + 0.01))?;
    
    chart.configure_mesh()
        .x_desc("Number of Active Features")
        .y_desc("Loss")
        .axis_desc_style(("sans-serif", 22).into_font())
        .x_label_style(("sans-serif", 20).into_font())
        .y_label_style(("sans-serif", 20).into_font())
        .draw()?;
    
    // Plot each unique point using the discrete color based on its frequency.
    for (key, ind) in unique_points {
        let frequency = *freq_map.get(&key).unwrap_or(&0);
        let x = ind.combination.iter().filter(|&&b| b).count() as i32;
        let y = ind.loss;
    
        // Determine which group the frequency falls into.
        let mut group_index = ((frequency as f64) / group_interval).floor() as usize;
        if group_index >= tick_count {
            group_index = tick_count - 1;
        }
        let color = discrete_colors[group_index];
    
        chart.draw_series(std::iter::once(
            Circle::new((x, y), 6, color.filled()),
        ))?;
    }
    
    // Draw the legend in the legend area using the same discrete groups.
    legend_area.fill(&WHITE)?;
    let legend_font = ("sans-serif", 20).into_font();
    let mut y_pos = 20;
    for i in 0..tick_count {
        let lower = (group_interval * i as f64).round() as usize;
        let upper = if i == tick_count - 1 {
            max_freq
        } else {
            (group_interval * (i + 1) as f64).round() as usize - 1
        };
        let label = format!("Freq: {} - {}", lower, upper);
        let color = discrete_colors[i];
        // Draw a small colored rectangle.
        legend_area.draw(&Rectangle::new(
            [(10, y_pos), (40, y_pos + 20)],
            color.filled(),
        ))?;
        // Draw the label vertically centered relative to the rectangle.
        legend_area.draw(&Text::new(
            label,
            (50, y_pos + 14),
            legend_font.clone(),
        ))?;
        y_pos += 40;
    }
    
    root.present()?;
    println!("Best Pareto Front overview plot saved to {}", filename);
    Ok(())
}

pub fn plot_histogram(population: &[MooCombination], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Group individuals by the fitness tuple from get_fitness.
    // For each group, store (loss, best_rank, frequency)
    let mut groups: HashMap<String, (f64, i8, usize)> = HashMap::new();
    for ind in population {
        if ind.loss.is_infinite() {
            continue;
        }
        // Use get_fitness to obtain (active features, loss)
        let (active, loss) = get_fitness(ind);
        let key = format!("({}, {:.3})", active, loss);
        groups
            .entry(key)
            .and_modify(|e: &mut (f64, i8, usize)| {
                e.1 = e.1.min(ind.rank);
                e.2 += 1;
            })
            .or_insert((loss, ind.rank, 1));
    }

    // Convert the groups into a vector and sort by the key (the fitness tuple as string).
    let mut group_vec: Vec<(String, f64, i8, usize)> = groups
        .into_iter()
        .map(|(key, (loss, rank, freq))| (key, loss, rank, freq))
        .collect();
    group_vec.sort_by(|a, b| a.0.cmp(&b.0));
    
    let num_groups = group_vec.len();
    if num_groups == 0 {
        println!("No valid groups to plot.");
        return Ok(());
    }

    // Find the maximum frequency among the groups (for the y-axis)
    let max_freq = group_vec.iter().map(|(_, _, _, freq)| *freq).max().unwrap();

    // Create the drawing area with 1000X600 pixels.
    let root = BitMapBackend::new(filename, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build the chart.
    let mut chart = ChartBuilder::on(&root)
        .caption("NSGA-II Final Population Histogram", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(0..num_groups as i32, 0..((max_freq + 1) as i32))?;


    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Fitness (Active Features, Loss)")
        .y_desc("Frequency")
        .x_label_formatter(&|&x| {
            if (x as usize) < group_vec.len() {
                format!("            {}", group_vec[x as usize].0)  // prepend spaces
            } else {
                "".to_string()
            }
        })
        .axis_desc_style(("sans-serif", 22).into_font())
        .x_label_style(("sans-serif", 20).into_font())
        .y_label_style(("sans-serif", 20).into_font())
        .draw()?;

    // Draw each group as a bar.
    for (i, (_key, _loss, rank, freq)) in group_vec.iter().enumerate() {
        let color = match rank {
            0 => &RED,
            1 => &MAGENTA,
            2 => &BLUE,
            3 => &GREEN,
            _ => &BLACK,
        };
        let x0 = i as i32;
        let x1 = x0 + 1;
        let y0 = 0;
        let y1 = *freq as i32;
    
        // Draw the filled rectangle.
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, y0), (x1, y1)],
                ShapeStyle::from(color).filled()
            )
        ))?;
        // Draw the border of the rectangle.
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, y0), (x1, y1)],
                ShapeStyle::from(&BLACK).stroke_width(1)
            )
        ))?;
    }

    root.present()?;
    println!("Histogram saved to {}", filename);
    Ok(())
}

pub fn init() {

    // 1) Loads the breast cancer dataset
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; 

    let population_size = 100; // Size of the population
    let generations = 1000; // Number of generations to run
    let generations_to_print = 100; // Print every x generations
    let mut gene_length = 9; // Number of features (columns) in the dataset
    let n_times = 100;

    // let final_sorted_population = run_moo_algorithm(file_path, population_size, generations, generations_to_print, gene_length);

    // // 8) Plot the final population
    // let plot_filename = "src/output/moo/nsga_2_breast_cancer.png";
    // plot_population(&final_sorted_population, plot_filename).unwrap();

    // // 2 LOads the wine dataset
    // let file_path = "XGB-Feature-Selection/output/wine_quality_combined";
    // let population_size = 100; // Size of the population
    // let generations = 1000; // Number of generations to run
    // let generations_to_print = 100; // Print every x generations
    // let gene_length = 11; // Number of features (columns) in the dataset

    // let final_sorted_population = run_moo_algorithm(file_path, population_size, generations, generations_to_print, gene_length);
    // // 8) Plot the final population
    // let plot_filename = "src/output/moo/nsga_2_wine_combined.png";
    // plot_population(&final_sorted_population, plot_filename).unwrap();

    // // 3) Plot the histogram for the final population
    // let histogram_filename = "src/output/moo/nsga_2_histogram.png";
    // plot_histogram(&final_sorted_population, histogram_filename).unwrap();
    // println!("Histogram saved to {}", histogram_filename);


    // 1) Loads the breast cancer dataset
    let file_path = "XGB-Feature-Selection/output/breast_cancer_wisconsin_original"; 
    let plot_filename = "src/output/moo/nsga_2_breast_cancer_multiple.png";
    let csv_filename = "src/output/moo/nsga_2_breast_cancer_summary.csv";
    gene_length = 9;

    // Starts a timer before the exec
    let mut start_time = Instant::now();

    let all_best_cancer = execute_run_n_times(
        n_times, 
        population_size, 
        generations, 
        generations_to_print, 
        gene_length, 
        file_path,
        false,
        true
    );
    // Ends the timer after the exec
    let mut elapsed_time = start_time.elapsed();
    println!("Elapsed time for the Breast Cancer dataset: {:?}", elapsed_time);

    plot_best_pareto_overview(&all_best_cancer, plot_filename, "Breast Cancer").unwrap();
    plot_histogram(&all_best_cancer, "src/output/moo/nsga_2_breast_cancer_histogram.png").unwrap();
    let _1 = save_run_metrics_to_csv(&all_best_cancer, csv_filename);

    // 2) Loads the wine quality dataset
    let file_path = "XGB-Feature-Selection/output/wine_quality_combined"; 
    let plot_filename = "src/output/moo/nsga_2_wine_combined_multiple.png";
    let csv_filename = "src/output/moo/nsga_2_wine_combined_summary.csv";
    gene_length = 11;

    // Reset the timer before the exec
    start_time = Instant::now();

    let all_best_wine = execute_run_n_times(
        n_times, 
        population_size, 
        generations, 
        generations_to_print, 
        gene_length, 
        file_path,
        false,
        true
    );
    // Ends the timer after the exec
    elapsed_time = start_time.elapsed();
    println!("Elapsed time for the Wine dataset: {:?}", elapsed_time);

    plot_best_pareto_overview(&all_best_wine, plot_filename, "Wine").unwrap();
    plot_histogram(&all_best_wine, "src/output/moo/nsga_2_wine_combined_histogram.png").unwrap();
    let _2 = save_run_metrics_to_csv(&all_best_wine, csv_filename);


    // 3) Loads the breast cancer dataset
    let file_path = "XGB-Feature-Selection/output/titanic"; 
    let plot_filename = "src/output/moo/nsga_2_titanic_multiple.png";
    let csv_filename = "src/output/moo/nsga_2_titanic_summary.csv";
    gene_length = 8;

    // Reset the timer before the exec
    start_time = Instant::now();

    let all_best_titanic = execute_run_n_times(
        n_times, 
        population_size, 
        generations, 
        generations_to_print, 
        gene_length, 
        file_path,
        false,
        true
    );

    // Ends the timer after the exec
    elapsed_time = start_time.elapsed();
    println!("Elapsed time for the Titanic dataset: {:?}", elapsed_time);

    plot_best_pareto_overview(&all_best_titanic, plot_filename, "Titanic").unwrap();
    plot_histogram(&all_best_titanic, "src/output/moo/nsga_2_titanic_histogram.png").unwrap();
    let _3 = save_run_metrics_to_csv(&all_best_titanic, csv_filename);



}