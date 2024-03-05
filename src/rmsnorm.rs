use tch::{Kind, Tensor, nn::{Module, Path}};
use crate::transformer::NormLayer;

/// Root Mean Square Layer Normalization
#[derive(Debug)]
pub struct RmsNorm {
    scale: Tensor,
    size: i64,
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let variance = (xs*xs).mean_dim(-1, true, Kind::BFloat16);
        let hidden_states = xs * (variance + 1e-5).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * hidden_states
    }
}

impl NormLayer for RmsNorm {
    fn new(p: &Path, size: i64) -> Self {
        let scale = p.zeros("weight", &[size]);
        Self { scale, size }
    }
}
