
use std::path;

use rand::Rng;
use crate::{structs::combination::Combination, utils::read_data::{read_data, read_test_data}};
use plotters::prelude::*;
const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 1000;
const TOURNAMENT_SIZE: usize = 3;
const W : f64 = 0.5;
const MAX_TIME : f64 = 1.0;
const PENALTY : f64 = 1.0/64.0;

/// Generates a population of candidate solutions represented as vectors of booleans.
///
/// Each solution in the population is a vector of `n` booleans generated at random.
/// The overall population size is determined by the constant `POPULATION_SIZE`.
///
/// # Parameters
///
/// * `n` - The number of boolean elements in each candidate solution.
///
/// # Returns
///
/// A vector containing `POPULATION_SIZE` candidate solutions, where each solution is a vector of `n` booleans.
///
pub fn generate_population(n: usize) -> Vec<Vec<bool>>{

    let mut rng = rand::rng();
    let mut population = Vec::new();
    for _ in 0..POPULATION_SIZE{
        let mut solution = Vec::new();
        for _ in 0..n{
            solution.push(rng.random_bool(0.5));
        }
        population.push(solution);
    }
    return population;
}

/// Calculates the cost function of a candidate solution.
/// 
/// The cost function is a weighted sum of the loss and the number of active features in the solution.
/// The loss is normalized by the largest loss in the dataset.
/// 
/// # Parameters
/// 
/// * `combinations` - A vector of `Combination` structs representing the dataset.
/// * `solution` - A vector of booleans representing the candidate solution.
/// 
/// # Returns
/// 
/// The cost of the candidate solution.
pub fn cost_function(combinations: &Vec<Combination>, solution: &Vec<bool>, largest_loss:f64, w: f64) -> f64 {
    let mut loss = get_loss(combinations, solution);
    loss = loss / largest_loss;
    let mut feature_count = 0;
    for feature in solution {
        if *feature {
            feature_count += 1;
        }
    }
    let feature_count = feature_count as f64 / solution.len() as f64;
    let cost = w * loss + (1.0 - w) * feature_count;
    return cost;
}


pub fn cost_function_test(combinations: &Vec<Combination>, solution: &Vec<bool>, penalty: f64) -> f64 {
    let loss = get_loss(combinations, solution);
    if loss.is_infinite() {
        return f64::MAX;
    }
    let mut feature_count = 0;
    for feature in solution {
        if *feature {
            feature_count += 1;
        }
    }
    return loss + penalty * feature_count as f64;
}

/// Selects a candidate solution from the population using tournament selection.
/// '
/// The tournament selection process involves selecting a random subset of solutions from the population
/// 
/// # Parameters
/// 
/// * `population` - A vector of candidate solutions.
/// * `combinations` - A vector of `Combination` structs representing the dataset.
/// * `largest_loss` - The largest loss in the dataset.
/// 
/// # Returns
/// 
/// The selected candidate solution.
pub fn tournament_selection(population: &Vec<Vec<bool>>, combinations: &Vec<Combination>, largest_loss:f64) -> Vec<bool> {
    let mut rng = rand::rng();
    let mut best_solution = population[rng.random_range(0..population.len())].clone();
    let mut best_cost = cost_function(combinations, &best_solution, largest_loss, W);
    for _ in 0..TOURNAMENT_SIZE {
        let solution = population[rng.random_range(0..population.len())].clone();
        let cost = cost_function(combinations, &solution, largest_loss, W);
        if cost < best_cost {
            best_solution = solution;
            best_cost = cost;
        }
    }
    return best_solution;
}


/// Performs single-point crossover between two parent solutions.
/// 
/// The crossover point is randomly selected, and the children are created by combining the first part of one parent with the second part of the other parent.
/// 
/// # Parameters
/// 
/// * `parent1` - The first parent solution.
/// * `parent2` - The second parent solution.
/// 
/// # Returns
/// 
/// A tuple containing the two child solutions.
pub fn single_point_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>) {
    let mut rng = rand::rng();
    let crossover_point = rng.random_range(0..parent1.len());
    let mut child1 = Vec::new();
    let mut child2 = Vec::new();
    for i in 0..parent1.len() {
        if i < crossover_point {
            child1.push(parent1[i]);
            child2.push(parent2[i]);
        } else {
            child1.push(parent2[i]);
            child2.push(parent1[i]);
        }
    }
    return (child1, child2);
}

/// Calculates the Hamming distance between two candidate solutions.
/// 
/// The Hamming distance is the number of positions at which the two solutions differ.
/// 
/// # Parameters
/// 
/// * `solution1` - The first candidate solution.
/// * `solution2` - The second candidate solution.
/// 
/// # Returns
/// 
/// The Hamming distance between the two solutions.
pub fn hamming_distance(solution1: &Vec<bool>, solution2: &Vec<bool>) -> usize {
    let mut distance = 0;
    for i in 0..solution1.len() {
        if solution1[i] != solution2[i] {
            distance += 1;
        }
    }
    return distance;
}


/// Performs bit-flip mutation on a candidate solution.
/// 
/// Each bit in the solution has a probability of being flipped, determined by the mutation rate.
/// 
/// # Parameters
/// 
/// * `solution` - The candidate solution to be mutated.
/// * `mutation_rate` - The probability of each bit being flipped.
pub fn bit_flip_mutation(solution: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::rng();
    for i in 0..solution.len() {
        if rng.random::<f64>() < mutation_rate {
            solution[i] = !solution[i];
        }
    }
}


/// Returns the loss of a candidate solution from the dataset.
/// 
/// # Parameters
/// 
/// * `combinations` - A vector of `Combination` structs representing the dataset.
/// * `solution` - The candidate solution.
/// 
/// # Returns
/// 
/// The loss of the candidate solution.
pub fn get_loss(combinations: &Vec<Combination>, solution: &Vec<bool>) -> f64 {
    for combination in combinations {
        if combination.combination == *solution {
            return combination.loss;
        }
    }
    return f64::MAX;
}


/// Runs the genetic algorithm to optimize the feature selection problem.
/// 
/// The genetic algorithm uses a population of candidate solutions, each represented as a vector of booleans.
/// 
/// # Parameters
/// 
/// * `combinations` - A vector of `Combination` structs representing the dataset.
/// 
/// # Returns
/// 
/// A tuple containing the best solution found, the cost of the best solution, and a vector of costs and losses over the generations.
pub fn genetic_algorithm(combinations: &Vec<Combination>) -> (Combination, f64, Vec<(f64, f64)>) {
    let n = combinations[0].combination.len();
    let mutation_rate = 1.0 / n as f64;
    let mut largest_loss = 0.0;
    for combination in combinations {
        if combination.loss > largest_loss {
            largest_loss = combination.loss;
        }
    }

    let mut population = generate_population(n);
    let mut costs = Vec::new();
    let mut best_solution: Combination = Combination { combination: population[0].clone(), loss: get_loss(&combinations, &population[0]) };
    let mut best_cost = cost_function_test(&combinations, &best_solution.combination, PENALTY);

    for solution in &population{
        let cost = cost_function_test(&combinations, solution, PENALTY);
        if cost < best_cost {
            best_solution.combination = solution.clone();
            best_solution.loss = get_loss(&combinations, solution);
            best_cost = cost;
        }
    }

    println!("Initial Best Cost: {}", best_cost);

    for gen in 0..MAX_GENERATIONS{
        let mut new_population = Vec::new();
        for i in 0..POPULATION_SIZE{
            let parent1 = tournament_selection(&population, &combinations, largest_loss);
            let parent2 = tournament_selection(&population, &combinations, largest_loss);
            let (mut child1, mut child2) = single_point_crossover(&parent1, &parent2);
            bit_flip_mutation(&mut child1, mutation_rate);
            bit_flip_mutation(&mut child2, mutation_rate);
            let distance1 = hamming_distance(&parent1, &child1);
            let distance2 = hamming_distance(&parent2, &child2);
            if distance1 < distance2 {
                if cost_function_test(&combinations, &child1, PENALTY) < cost_function_test(&combinations, &parent1, PENALTY) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            } else {
                if cost_function_test(&combinations, &child2,  PENALTY) < cost_function_test(&combinations, &parent2, PENALTY) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            }
        }

        population = new_population;

        for solution in &population{
            let cost = cost_function_test(&combinations, solution, PENALTY);
            if cost < best_cost {
                best_solution.combination = solution.clone();
                best_solution.loss = get_loss(&combinations, solution);
                best_cost = cost;
            }
        }
        costs.push((best_cost,best_solution.loss));

        if gen % 10 == 0 {
            println!("Generation: {}, Best Cost: {}", gen, best_cost);
        }
    }

    println!("Final Best Cost: {}", best_cost);
    costs.push((best_cost,best_solution.loss));
    return (best_solution, best_cost, costs);

}


/// Runs the genetic algorithm to optimize the feature selection problem with a time limit.
/// 
/// The genetic algorithm uses a population of candidate solutions, each represented as a vector of booleans.
/// 
/// # Parameters
/// 
/// * `combinations` - A vector of `Combination` structs representing the dataset.
/// * `time_limit_secs` - The time limit in seconds.
/// 
/// # Returns
/// 
/// A tuple containing the best solution found, the cost of the best solution, a vector of costs and losses over the generations, and the number of generations.
pub fn genetic_algorithm_time(
    combinations: &Vec<Combination>,
    time_limit_secs: f64
) -> (Combination, f64, Vec<(f64, f64)>, usize) {
    let n = combinations[0].combination.len();
    let mutation_rate = 1.0 / n as f64;
    let mut largest_loss = 0.0;
    for combination in combinations {
        if combination.loss > largest_loss {
            largest_loss = combination.loss;
        }
    }

    let mut population = generate_population(n);
    let mut costs = Vec::new();
    let mut best_solution: Combination = Combination {
        combination: population[0].clone(),
        loss: get_loss(&combinations, &population[0]),
    };
    let mut best_cost = cost_function(&combinations, &best_solution.combination, largest_loss, W);
    for solution in &population {
        let cost = cost_function(&combinations, solution, largest_loss, W);
        if cost < best_cost {
            best_solution.combination = solution.clone();
            best_solution.loss = get_loss(&combinations, solution);
            best_cost = cost;
        }
    }

    println!("Initial Best Cost: {}", best_cost);

    let start_time = std::time::Instant::now();
    let generation: usize = 0;
    let mut best_generation: usize = 0;

    while start_time.elapsed().as_secs_f64() < time_limit_secs {
        let mut new_population = Vec::new();
        for _ in 0..POPULATION_SIZE {
            let parent1 = tournament_selection(&population, &combinations, largest_loss);
            let parent2 = tournament_selection(&population, &combinations, largest_loss);
            let (mut child1, mut child2) = single_point_crossover(&parent1, &parent2);
            bit_flip_mutation(&mut child1, mutation_rate);
            bit_flip_mutation(&mut child2, mutation_rate);
            let distance1 = hamming_distance(&parent1, &child1);
            let distance2 = hamming_distance(&parent2, &child2);
            if distance1 < distance2 {
                if cost_function(&combinations, &child1, largest_loss, W)
                    < cost_function(&combinations, &parent1, largest_loss, W)
                {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            } else {
                if cost_function(&combinations, &child2, largest_loss, W)
                    < cost_function(&combinations, &parent2, largest_loss, W)
                {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            }
        }

        population = new_population;
        for solution in &population {
            let cost = cost_function(&combinations, solution, largest_loss, W);
            if cost < best_cost {
                best_solution.combination = solution.clone();
                best_solution.loss = get_loss(&combinations, solution);
                best_cost = cost;
                best_generation = generation+1;
            }
        }
        costs.push((best_cost, best_solution.loss));
    }
    println!("Final Best Cost: {}", best_cost);

    costs.push((best_cost, best_solution.loss));
    (best_solution, best_cost, costs, best_generation)
}


/// Plots the costs and losses over the generations.
/// 
/// # Parameters
/// 
/// * `cost_loss` - A vector of tuples containing the cost and loss values over the generations.
/// * `filename` - The name of the output file.
fn plot_costs_and_loss(cost_loss: &Vec<(f64, f64)>, filename: &str) {
    use plotters::prelude::*;

    let root_area = BitMapBackend::new(filename, (800, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let generations = cost_loss.len();
    let x_range = 0..generations;

    let mut y_min = std::f64::MAX;
    let mut y_max = std::f64::MIN;
    for (cost, loss) in cost_loss.iter() {
        y_min = y_min.min(*cost).min(*loss);
        y_max = y_max.max(*cost).max(*loss);
    }

    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.01;
        y_max += 0.01;
    } else {
        let margin = (y_max - y_min) * 0.1;
        y_min -= margin;
        y_max += margin;
    }

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Cost and Loss Over Generations", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_min..y_max)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Generation")
        .y_desc("Value")
        .draw()
        .unwrap();

    chart.draw_series(LineSeries::new(
        cost_loss.iter().enumerate().map(|(i, &(cost, _))| (i, cost)),
        &RED,
    )).unwrap()
      .label("Cost")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));


    chart.draw_series(LineSeries::new(
        cost_loss.iter().enumerate().map(|(i, &(_, loss))| (i, loss)),
        &BLUE,
    )).unwrap()
      .label("Loss")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    root_area.present().unwrap();
}

/// Plots a histogram of the population.
///
/// The histogram shows the distribution of the number of active features and the loss values in the population.
/// 
/// # Parameters
/// 
/// * `population` - A vector of `Combination` structs representing the population.
/// * `filename` - The name of the output file.
/// 
/// # Returns
/// 
/// A `Result` containing `Ok(())` if the histogram was successfully saved, or an error message if an error occurred.
pub fn plot_histogram(population: &[Combination], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    let mut groups: HashMap<String, (f64, i8, usize)> = HashMap::new();
    for ind in population {
        if ind.loss.is_infinite() {
            continue;
        }
        let active = ind.combination.iter().filter(|&&b| b).count();
        let loss = ind.loss;
        let key = format!("({}, {:.3})", active, loss);
        groups
        .entry(key)
        .and_modify(|e: &mut (f64, i8, usize)| {
            e.1 = 0;
            e.2 += 1;
        })
        .or_insert((loss, 0, 1));
        
    }

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

    let max_freq = group_vec.iter().map(|(_, _, _, freq)| *freq).max().unwrap();
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Individuals histogram", ("sans-serif", 30))
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
                format!("                    {}", group_vec[x as usize].0)  // prepend spaces
            } else {
                "".to_string()
            }
        })
        .axis_desc_style(("sans-serif", 22).into_font())
        .x_label_style(("sans-serif", 20).into_font())
        .y_label_style(("sans-serif", 20).into_font())
        .draw()?;

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
    
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, y0), (x1, y1)],
                ShapeStyle::from(color).filled()
            )
        ))?;
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


/// Optimizes the feature selection problem using the single output optimization technique.
/// 
/// The single output optimization technique involves running the genetic algorithm on three datasets:
/// - breast_cancer_wisconsin_original
/// - titanic
/// - wine_quality_combined
/// 
/// The best solution, best cost, and costs over the generations are printed to the console.
/// 
/// 
/// # Parameters
/// 
/// * `path_vec` - An array of strings representing the paths to the datasets.
/// 
/// # Returns
/// 
/// A tuple containing the best solution found, the cost of the best solution, and a vector of costs and losses over the generations.
pub(crate) fn single_output_optimization(path_vec: [&str; 3]) {

    for path in path_vec.iter() {
        let combinations = read_data(&format!("XGB-Feature-Selection/output/{}", path)).unwrap();
        let (best_solution, best_cost, costs) = genetic_algorithm(&combinations);
        println!("Best Solution: {:?}, Best Cost: {}", best_solution.combination, best_cost);
        // Create the output file path using format! macro.
        let output_path = format!("output/soo/{}.png", path);
        plot_costs_and_loss(&costs, &output_path);
    }

}

pub (crate) fn single_output_test_optimization(path: &str){
    let combinations = read_test_data(&format!("XGB-Feature-Selection/output_test/{}", path)).unwrap();
        let (best_solution, best_cost, costs) = genetic_algorithm(&combinations);
        println!("Best Solution: {:?}, Best Cost: {}", best_solution.combination, best_cost);
}


/// Optimizes the feature selection problem using the single output optimization technique with a time limit.
/// 
/// The single output optimization technique involves running the genetic algorithm on three datasets:
/// 
/// - breast_cancer_wisconsin_original
/// - titanic
/// - wine_quality_combined
/// 
/// The best solution, best cost, costs over the generations, and the number of generations are printed to the console.
/// 
/// # Parameters
/// 
/// * `path_vec` - An array of strings representing the paths to the datasets.
/// 
/// # Returns
/// 
/// A tuple containing the best solution found, the cost of the best solution, a vector of costs and losses over the generations, and the number of generations.
pub(crate) fn multi_run_validation(path: &str){
    let combinations = read_data(&format!("XGB-Feature-Selection/output/{}", path)).unwrap();
    let mut best_solutions = Vec::new();
    let mut costs = Vec::new();
    let mut feauter_counts = Vec::new();
    let mut losses = Vec::new();
    let mut best_generations = Vec::new();
    for _ in 0..100{
        let (solution, cost, _, best_generation) = genetic_algorithm_time(&combinations, MAX_TIME);
        best_solutions.push(solution.clone());
        costs.push(cost);
        let mut feature_count = 0;
        for feature in &solution.combination {
            if *feature {
                feature_count += 1;
            }
        }
        feauter_counts.push(feature_count);
        losses.push(solution.loss);
        best_generations.push(best_generation*POPULATION_SIZE);

    }
    let mut mean_best_cost = 0.0;
    let mut mean_feature_count = 0.0;
    let mut mean_loss = 0.0;
    let mut mean_evaluation_number = 0.0;
    let mut mean_best_solution = best_solutions[0].clone();

    for cost in &costs {
        mean_best_cost += *cost;
    }
    mean_best_cost /= costs.len() as f64;
    for feature_count in &feauter_counts {
        mean_feature_count += *feature_count as f64;
    }
    mean_feature_count /= feauter_counts.len() as f64;
    for loss in &losses {
        mean_loss += *loss;
    }
    mean_loss /= losses.len() as f64;
    for evaluation_number in &best_generations {
        mean_evaluation_number += *evaluation_number as f64;
    }
    mean_evaluation_number /= best_generations.len() as f64;
    let mut max_count = 0;
    for solution in &best_solutions {
        let count = best_solutions.iter().filter(|&s| s == solution).count();
        if count > max_count {
            max_count = count;
            mean_best_solution = solution.clone();
        }
    }

    mean_best_cost /= best_solutions.len() as f64;
    println!("Mean Best Cost: {}", mean_best_cost);
    println!("Mean Feature Count: {}", mean_feature_count);
    println!("Mean Loss: {}", mean_loss);
    println!("Mean Evaluation Number: {}", mean_evaluation_number);
    let output_path = format!("src/output/soo/{}_histogram.png", path);
    plot_histogram(&best_solutions, &output_path).unwrap();
    let csv_path = format!("src/output/soo/{}.csv", path);
    save_data_to_csv(&csv_path, mean_feature_count, mean_loss, mean_best_cost, mean_evaluation_number,mean_best_solution);
    
}


/// Saves the mean feature count, mean loss, mean best cost, mean evaluation number, and mean best solution to a CSV file.
/// 
/// # Parameters
/// 
/// * `path` - The path to the output CSV file.
/// * `mean_feature_count` - The mean number of active features in the best solutions.
/// * `mean_loss` - The mean loss of the best solutions.
/// * `mean_best_cost` - The mean cost of the best solutions.
/// * `mean_evaluation_number` - The mean number of evaluations required to find the best solutions.
/// * `mean_best_solution` - The mean best solution found.
/// 
fn save_data_to_csv(path: &str, mean_feature_count: f64, mean_loss: f64, mean_best_cost: f64, mean_evaluation_number: f64, mean_best_solution: Combination){
    let mut wtr = csv::Writer::from_path(path).unwrap();
    wtr.write_record(&["Mean Feature Count", "Mean Loss", "Mean Best Cost", "Mean Evaluation Number", "Mean Best Solution"]).unwrap();
    wtr.write_record(&[mean_feature_count.to_string(), mean_loss.to_string(), mean_best_cost.to_string(), mean_evaluation_number.to_string(), mean_best_solution.to_string()]).unwrap();
    wtr.flush().unwrap();


}

