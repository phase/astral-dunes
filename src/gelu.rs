use tch::{Tensor, nn::{Path, Module}};
use crate::{linear::Linear, transformer::FeedForward};

/// FFN using Gaussian Error Linear Units, activation function
#[derive(Debug)]
pub struct Gelu {
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
}

impl Module for Gelu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.gate_proj).silu().apply(&self.down_proj) * xs.apply(&self.up_proj)
    }
}

impl FeedForward for Gelu {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64) -> Self {
        let gate_proj = Linear::new_no_bias(p / "gate_proj", hidden_dim, in_dim);
        let down_proj = Linear::new_no_bias(p / "down_proj", in_dim, hidden_dim);
        let up_proj = Linear::new_no_bias(p / "up_proj", hidden_dim, in_dim);

        Self {
            down_proj,
            gate_proj,
            up_proj,
        }
    }
}
