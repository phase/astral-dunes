use tch::{Tensor, nn::{Path, Module}};
use crate::{linear::Linear, transformer::FeedForward};

/// FFN using the Sigmoid Linear Unit (aka Swish) activation function
#[derive(Debug)]
pub struct Swish {
    lin1: Linear,
    lin2: Linear,
}

impl Module for Swish {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // silu(x) = x * sigmoid(x)
        xs.apply(&self.lin1).silu().apply(&self.lin2)
    }
}

impl FeedForward for Swish {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
        let lin1 = Linear::new_no_bias(p / "lin1", in_dim, hidden_dim);
        let lin2 = Linear::new_no_bias(p / "lin2", hidden_dim, out_dim);
        Self { lin1, lin2 }
    }
}
