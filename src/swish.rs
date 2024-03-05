use tch::{Tensor, nn::{Path, Module}};
use crate::{linear::Linear, transformer::FeedForward};

/// FFN using the Sigmoid Linear Unit (aka Swish) activation function
#[derive(Debug)]
pub struct Swish {
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
}

impl Module for Swish {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // silu = x * sigmoid(x)
        xs.apply(&self.gate_proj).silu().apply(&self.down_proj) * xs.apply(&self.up_proj)
    }
}

impl FeedForward for Swish {
    fn new(p: &Path, in_dim: i64, out_dim: i64) -> Self {
        let gate_proj = Linear::new_no_bias(p / "gate_proj", out_dim, in_dim);
        let down_proj = Linear::new_no_bias(p / "down_proj", in_dim, out_dim);
        let up_proj = Linear::new_no_bias(p / "up_proj", out_dim, in_dim);

        Self {
            down_proj,
            gate_proj,
            up_proj,
        }
    }
}
