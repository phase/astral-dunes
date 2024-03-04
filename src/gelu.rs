use tch::{Tensor, nn::{Path, Module}};
use crate::{linear::Linear, transformer::FeedForward};

/// FFN using gelu activation function
#[derive(Debug)]
pub struct Gelu {
    lin1: Linear,
    lin2: Linear,
}

impl Module for Gelu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.lin1).gelu("none").apply(&self.lin2)
    }
}

impl FeedForward for Gelu {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
        let lin1 = Linear::new(p / "lin1", in_dim, hidden_dim);
        let lin2 = Linear::new(p / "lin2", hidden_dim, out_dim);
        Self { lin1, lin2 }
    }
}
