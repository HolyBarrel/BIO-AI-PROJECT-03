
use std::path;

use rand::Rng;
use crate::{structs::combination::Combination, utils::read_data::read_data};
use plotters::prelude::*;
const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 1000;
const TOURNAMENT_SIZE: usize = 3;
const W : f64 = 0.5;
const MAX_TIME : f64 = 5.0;

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

pub fn hamming_distance(solution1: &Vec<bool>, solution2: &Vec<bool>) -> usize {
    let mut distance = 0;
    for i in 0..solution1.len() {
        if solution1[i] != solution2[i] {
            distance += 1;
        }
    }
    return distance;
}

pub fn bit_flip_mutation(solution: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::rng();
    for i in 0..solution.len() {
        if rng.random::<f64>() < mutation_rate {
            solution[i] = !solution[i];
        }
    }
}

pub fn get_loss(combinations: &Vec<Combination>, solution: &Vec<bool>) -> f64 {
    for combination in combinations {
        if combination.combination == *solution {
            return combination.loss;
        }
    }
    return f64::MAX;
}

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
    let mut best_cost = cost_function(&combinations, &best_solution.combination, largest_loss, W);

    for solution in &population{
        let cost = cost_function(&combinations, solution, largest_loss, W);
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
                if cost_function(&combinations, &child1, largest_loss,W) < cost_function(&combinations, &parent1, largest_loss,W) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            } else {
                if cost_function(&combinations, &child2, largest_loss, W) < cost_function(&combinations, &parent2, largest_loss, W) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            }
        }

        population = new_population;

        for solution in &population{
            let cost = cost_function(&combinations, solution, largest_loss, W);
            if cost < best_cost {
                best_solution.combination = solution.clone();
                best_solution.loss = get_loss(&combinations, solution);
                best_cost = cost;
            }
        }
        costs.push((best_cost,best_solution.loss));
    }

    println!("Final Best Cost: {}", best_cost);
    costs.push((best_cost,best_solution.loss));
    return (best_solution, best_cost, costs);

}

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
    // Initialize best_solution with the first individual in the population
    let mut best_solution: Combination = Combination {
        combination: population[0].clone(),
        loss: get_loss(&combinations, &population[0]),
    };
    let mut best_cost = cost_function(&combinations, &best_solution.combination, largest_loss, W);

    // Check the initial population for the best solution
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
    let mut generation: usize = 0;
    let mut best_generation: usize = 0;
    
    // Run until the elapsed time exceeds the given time limit.
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

        // Update the best solution from the new population.
        for solution in &population {
            let cost = cost_function(&combinations, solution, largest_loss, W);
            if cost < best_cost {
                best_solution.combination = solution.clone();
                best_solution.loss = get_loss(&combinations, solution);
                best_cost = cost;
                best_generation = generation;
            }
        }
        costs.push((best_cost, best_solution.loss));

        if generation % 1000 == 0 {
            println!("Generation: {}, Best Cost: {}", generation, best_cost);
        }
        generation += 1;
    }

    // Final push (optional, depending on your logging preference)
    costs.push((best_cost, best_solution.loss));
    (best_solution, best_cost, costs, best_generation)
}


fn plot_costs_and_loss(cost_loss: &Vec<(f64, f64)>, filename: &str) {
    use plotters::prelude::*;

    // Create a drawing area and fill it with white.
    let root_area = BitMapBackend::new(filename, (800, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    // x-axis range is the number of generations.
    let generations = cost_loss.len();
    let x_range = 0..generations;

    // Determine y-axis range based on cost and loss values.
    let mut y_min = std::f64::MAX;
    let mut y_max = std::f64::MIN;
    for (cost, loss) in cost_loss.iter() {
        y_min = y_min.min(*cost).min(*loss);
        y_max = y_max.max(*cost).max(*loss);
    }

    // If the range is too narrow, manually expand it.
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.01;
        y_max += 0.01;
    } else {
        let margin = (y_max - y_min) * 0.1;
        y_min -= margin;
        y_max += margin;
    }

    // Build the chart.
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

    // Plot the "Cost" line in red.
    chart.draw_series(LineSeries::new(
        cost_loss.iter().enumerate().map(|(i, &(cost, _))| (i, cost)),
        &RED,
    )).unwrap()
      .label("Cost")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot the "Loss" line in blue.
    chart.draw_series(LineSeries::new(
        cost_loss.iter().enumerate().map(|(i, &(_, loss))| (i, loss)),
        &BLUE,
    )).unwrap()
      .label("Loss")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Configure and draw the legend.
    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    // Save the drawing.
    root_area.present().unwrap();
}

pub fn plot_histogram(population: &[Combination], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    // Group individuals by the fitness tuple from get_fitness.
    // For each group, store (loss, best_rank, frequency)
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

    // Find the maximum frequency among the groups (for the y-axis)
    let max_freq = group_vec.iter().map(|(_, _, _, freq)| *freq).max().unwrap();

    // Create the drawing area with 800x600 pixels.
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build the chart.
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


pub(crate) fn multi_run_validation(path: &str){
    let combinations = read_data(&format!("XGB-Feature-Selection/output/{}", path)).unwrap();
    let mut best_solutions = Vec::new();
    let mut costs = Vec::new();
    let mut feauter_counts = Vec::new();
    let mut losses = Vec::new();
    let mut best_generations = Vec::new();
    for _ in 0..10{
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

    mean_best_cost /= best_solutions.len() as f64;
    println!("Mean Best Cost: {}", mean_best_cost);
    println!("Mean Feature Count: {}", mean_feature_count);
    println!("Mean Loss: {}", mean_loss);
    println!("Mean Evaluation Number: {}", mean_evaluation_number);
    let output_path = format!("src/output/soo/{}_histogram.png", path);
    plot_histogram(&best_solutions, &output_path).unwrap();
    let csv_path = format!("src/output/soo/{}.csv", path);
    save_data_to_csv(&csv_path, costs.len(), mean_feature_count, mean_loss, mean_best_cost, mean_evaluation_number);
    
}


fn save_data_to_csv(path: &str,run_counts:usize, mean_feature_count: f64, mean_loss: f64, mean_best_cost: f64, mean_evaluation_number: f64){
    let mut wtr = csv::Writer::from_path(path).unwrap();
    wtr.write_record(&["Run Counts","Mean Feature Count", "Mean Loss", "Mean Best Cost", "Mean Evaluation Number"]).unwrap();
    wtr.write_record(&[run_counts.to_string() ,mean_feature_count.to_string(), mean_loss.to_string(), mean_best_cost.to_string(), mean_evaluation_number.to_string()]).unwrap();
    wtr.flush().unwrap();


}

