#[derive(Debug, Clone, Copy)]
pub enum UpdateMode {
    Global,
    KNeighbor(usize),
}