mod utils;
mod structs;

fn main() {
    let file_path = "XGB-Feature-Selection/output/titanic";
    let combinations = utils::read_data::read_data(file_path).unwrap();
    println!("{:?}", combinations);
}
