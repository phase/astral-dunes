use tch::{nn::{Module, Path}, Kind, Tensor};
use crate::{linear::Linear, transformer::FeedForward};

/// FFN using the Sigmoid Linear Unit (aka Swish) activation function
#[derive(Debug)]
pub struct Swish {
    /// aka w1
    gate_proj: Linear,
    /// aka w2
    down_proj: Linear,
    /// aka w3
    up_proj: Linear,
}

impl Module for Swish {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // silu = x * sigmoid(x)
        let xs = xs.apply(&self.gate_proj).silu() * xs.apply(&self.up_proj);
        xs.apply(&self.down_proj)
    }
}

impl FeedForward for Swish {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, kind: Kind) -> Self {
        let gate_proj = Linear::new_no_bias(p / "gate_proj", in_dim, hidden_dim, kind);
        let down_proj = Linear::new_no_bias(p / "down_proj", hidden_dim, in_dim, kind);
        let up_proj = Linear::new_no_bias(p / "up_proj", in_dim, hidden_dim, kind);

        Self {
            down_proj,
            gate_proj,
            up_proj,
        }
    }
}
