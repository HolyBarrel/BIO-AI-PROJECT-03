
/// Represents a combination of features and its loss
#[derive(Debug, Clone, PartialEq)]
pub struct Combination {
    pub combination: Vec<bool>,
    pub loss: f64,
}



impl Combination {

    /// Creates a new Combination
    /// 
    /// # Arguments
    /// 
    /// * `combination` - A vector of booleans representing the features in the combination
    /// * `loss` - The loss of the combination
    /// 
    /// # Returns
    /// 
    /// A new Combination
    pub fn new(combination: Vec<bool>, loss: f64) -> Self {
        Combination {
            combination,
            loss,
        }
    }

    /// Returns the number of features in the combination
    /// 
    /// # Returns
    /// 
    /// A `String` representing the combination as a sequence of '1's and '0's.
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