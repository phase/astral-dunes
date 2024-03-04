use tch::{Tensor, nn::{Module, Path}};
use crate::{WEIGHT_DECAY_GROUP, NO_WEIGHT_DECAY_GROUP};

#[derive(Debug)]
pub struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    pub fn new(p: Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = p.set_group(WEIGHT_DECAY_GROUP);
        let no_wd = p.set_group(NO_WEIGHT_DECAY_GROUP);
        Self {
            // x{in_dim} * w{out_dim, in_dim} + b{in_dim}
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
            bs: no_wd.zeros("bias", &[out_dim]),
        }
    }

    pub fn new_no_bias(p: Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = p.set_group(WEIGHT_DECAY_GROUP);
        let no_wd = p.set_group(NO_WEIGHT_DECAY_GROUP);
        Self {
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
            bs: no_wd.zeros_no_train("bias", &[out_dim]),
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}
