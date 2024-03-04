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
        let norm = (xs*xs).mean_dim(-1, true, Kind::Float);
        let xs_norm = xs * (norm + 1e-5).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * xs_norm
    }
}

impl NormLayer for RmsNorm {
    fn new(p: &Path, size: i64) -> Self {
        let scale = p.zeros("scale", &[size]);
        Self { scale, size }
    }
}
