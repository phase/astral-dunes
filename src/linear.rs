use crate::{NO_WEIGHT_DECAY_GROUP, WEIGHT_DECAY_GROUP};
use tch::{
    nn::{Module, Path},
    Kind, Tensor,
};

#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    bs: Option<Tensor>,
}

impl Linear {
    pub fn _new(p: Path, in_dim: i64, out_dim: i64, kind: Kind) -> Self {
        let wd = p.set_group(WEIGHT_DECAY_GROUP);
        let no_wd = p.set_group(NO_WEIGHT_DECAY_GROUP);
        Self {
            // x{in_dim} * w{out_dim, in_dim} + b{in_dim}
            ws: wd
                .randn("weight", &[out_dim, in_dim], 0.0, 0.02)
                .to_dtype(kind, false, false),
            bs: Some(no_wd.zeros("bias", &[out_dim]).to_dtype(kind, false, false)),
        }
    }

    pub fn new_no_bias(p: Path, in_dim: i64, out_dim: i64, kind: Kind) -> Self {
        let wd = p.set_group(WEIGHT_DECAY_GROUP);
        Self {
            ws: wd
                .randn("weight", &[out_dim, in_dim], 0.0, 0.02)
                .to_dtype(kind, false, false),
            bs: None,
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if let Some(bias) = &self.bs {
            xs.matmul(&self.ws.tr()) + bias
        } else {
            xs.matmul(&self.ws.tr())
        }
    }
}
