use rand::Rng;
const POPULATION_SIZE: usize = 100;

pub fn generate_population() -> Vec<bool>{
    let mut rng = rand::thread_rng();
    let mut population: Vec<bool> = Vec::new();
    for _ in 0..POPULATION_SIZE {
        population.push(rng.gen_bool(0.5));
    }
}