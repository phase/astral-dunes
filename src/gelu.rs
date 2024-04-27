use crate::{linear::Linear, transformer::FeedForward};
use tch::{
    nn::{Module, Path},
    Kind, Tensor,
};

/// FFN using Gaussian Error Linear Units, activation function
#[derive(Debug)]
pub struct Gelu {
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
}

impl Module for Gelu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.gate_proj).gelu("none") * xs.apply(&self.up_proj);
        xs.apply(&self.down_proj)
    }
}

impl FeedForward for Gelu {
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
