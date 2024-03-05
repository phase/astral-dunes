use tch::{Tensor, IndexOp, nn::{Path, ModuleT}};
use crate::{linear::Linear, Config};
use crate::transformer::AttentionLayer;

// Causal Self Attention
#[derive(Debug)]
pub struct SelfAttention {
    n_head: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    key: Linear,
    query: Linear,
    value: Linear,
    proj: Linear,
    mask: Tensor,
}

impl AttentionLayer for SelfAttention {
    fn new(p: &Path, cfg: &Config) -> Self {
        let key = Linear::new(p / "k_proj", cfg.n_embd, cfg.n_embd);
        let query = Linear::new(p / "q_proj", cfg.n_embd, cfg.n_embd);
        let value = Linear::new(p / "v_proj", cfg.n_embd, cfg.n_embd);
        let proj = Linear::new(p / "o_proj", cfg.n_embd, cfg.n_embd);
        let mask_init = Tensor::ones(
            [cfg.block_size, cfg.block_size],
            (tch::Kind::Float, p.device()),
        ).tril(0);
        let mask = mask_init.view([1, 1, cfg.block_size, cfg.block_size]);
        Self {
            n_head: cfg.n_head,
            attn_pdrop: cfg.attn_pdrop,
            resid_pdrop: cfg.resid_pdrop,
            key,
            query,
            value,
            proj,
            mask,
        }
    }
}

impl ModuleT for SelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // Batch size, sequence length, embedding dimensionality (n_embed)
        let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
        let head_size = sz_c / self.n_head;
        let sizes = [sz_b, sz_t, self.n_head, head_size];

        let (k, q, v) = (
            xs.apply(&self.key).view(sizes).transpose(1, 2),
            xs.apply(&self.query).view(sizes).transpose(1, 2),
            xs.apply(&self.value).view(sizes).transpose(1, 2),
        );

        // Embedding goes here

        // Causaul Self Attention
        let attn = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(head_size as f64));
        let attn = attn
            .masked_fill(&self.mask.i((.., .., ..sz_t, ..sz_t)).eq(0.), f64::NEG_INFINITY)
            .softmax(-1, tch::Kind::Float)
            .dropout(self.attn_pdrop, train);
        let ys = attn.matmul(&v);

        // reassembly head outputs side by side
        let ys = ys.transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);

        // output projection
        ys.apply(&self.proj).dropout(self.resid_pdrop, train)
    }
}
