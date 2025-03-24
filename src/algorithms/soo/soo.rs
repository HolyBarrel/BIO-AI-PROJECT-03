
use std::path;

use rand::Rng;
use crate::{structs::combination::Combination, utils::read_data::read_data};
use plotters::prelude::*;
const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 10000;
const TOURNAMENT_SIZE: usize = 3;

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

pub fn cost_function(combinations: &Vec<Combination>, solution: &Vec<bool>) -> f64 {
    let mut feature_count = 0;
    let mut cost = 0.0;
    for feature in solution {
        if *feature {
            feature_count += 1;
        }
    }
    if feature_count == 0 {
        return f64::MAX;
    }
    
    cost = get_loss(combinations, solution);

    return cost + (feature_count as f64 / (10.0 * solution.len() as f64) as f64);
}

pub fn tournament_selection(population: &Vec<Vec<bool>>, combinations: &Vec<Combination>) -> Vec<bool> {
    let mut rng = rand::rng();
    let mut best_solution = population[rng.random_range(0..population.len())].clone();
    let mut best_cost = cost_function(combinations, &best_solution);
    for _ in 0..TOURNAMENT_SIZE {
        let solution = population[rng.random_range(0..population.len())].clone();
        let cost = cost_function(combinations, &solution);
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

    let mut population = generate_population(n);
    let mut costs = Vec::new();
    let mut best_solution: Combination = Combination {combination: Vec::new(), loss: f64::MAX};
    let mut best_cost = cost_function(&combinations, &best_solution.combination);

    for solution in &population{
        let cost = cost_function(&combinations, solution);
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
            let parent1 = tournament_selection(&population, &combinations);
            let parent2 = tournament_selection(&population, &combinations);
            let (mut child1, mut child2) = single_point_crossover(&parent1, &parent2);
            bit_flip_mutation(&mut child1, mutation_rate);
            bit_flip_mutation(&mut child2, mutation_rate);
            let distance1 = hamming_distance(&parent1, &child1);
            let distance2 = hamming_distance(&parent2, &child2);
            if distance1 < distance2 {
                if cost_function(&combinations, &child1) < cost_function(&combinations, &parent1) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            } else {
                if cost_function(&combinations, &child2) < cost_function(&combinations, &parent2) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            }
        }

        population = new_population;

        for solution in &population{
            let cost = cost_function(&combinations, solution);
            if cost < best_cost {
                best_solution.combination = solution.clone();
                best_solution.loss = get_loss(&combinations, solution);
                best_cost = cost;
            }
        }
        costs.push((best_cost,best_solution.loss));
        

        if gen % 1000 == 0 {
            println!("Generation: {}, Best Cost: {}", gen, best_cost);
        }
    }

    costs.push((best_cost,best_solution.loss));
    return (best_solution, best_cost, costs);

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
