

#[derive(Debug, Clone, PartialEq)]
pub struct Combination {
    pub combination: Vec<bool>,
    pub loss: f64,
}


impl Combination {
    pub fn new(combination: Vec<bool>, loss: f64) -> Self {
        Combination {
            combination,
            loss,
        }
    }

    pub fn to_string(&self) -> String {
        let mut result = String::new();
        for i in 0..self.combination.len() {
            if self.combination[i] {
                result.push_str("1");
            } else {
                result.push_str("0");
            }
        }
        result
    }
}