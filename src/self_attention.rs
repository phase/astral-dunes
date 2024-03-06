use tch::{Tensor, IndexOp, nn::{Path, ModuleT}};
use crate::{linear::Linear, Config};
use crate::transformer::AttentionLayer;

// Causal Self Attention
#[derive(Debug)]
pub struct SelfAttention {
    n_head: i64,
    n_kv_heads: i64,
    head_dim: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    mask: Tensor,
}

impl AttentionLayer for SelfAttention {
    fn new(p: &Path, cfg: &Config) -> Self {
        let query = Linear::new(p / "q_proj", cfg.dim, cfg.n_head * cfg.head_dim, cfg.kind);
        let key = Linear::new(p / "k_proj", cfg.dim, cfg.n_kv_heads * cfg.head_dim, cfg.kind);
        let value = Linear::new(p / "v_proj", cfg.dim, cfg.n_kv_heads * cfg.head_dim, cfg.kind);
        let proj = Linear::new(p / "o_proj", cfg.n_kv_heads * cfg.head_dim, cfg.dim, cfg.kind);
        let mask_init = Tensor::ones(
            [cfg.block_size, cfg.block_size],
            (cfg.kind, p.device()),
        ).tril(0);
        let mask = mask_init.view([1, 1, cfg.block_size, cfg.block_size]);
        Self {
            n_head: cfg.n_head,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            attn_pdrop: cfg.attn_pdrop,
            resid_pdrop: cfg.resid_pdrop,
            key,
            query,
            value,
            proj,
            mask
        }
    }
}

impl ModuleT for SelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // Batch size, sequence length, embedding dimensionality (n_embed)
        let (B, L, D) = xs.size3().unwrap();
        let repeats = self.n_head / self.n_kv_heads;
        let head_size = D / self.n_head;

        let qsize = [B * L, self.n_head, self.head_dim];
        let sizes = [B * L, self.n_kv_heads, self.head_dim];

        let (q, k, v) = (
            xs.apply(&self.query).view(qsize),
            xs.apply(&self.key).view(sizes),
            xs.apply(&self.value).view(sizes),
        );

        // Rope Embedding goes here

        // Repeat keys and values to match the number of query heads
        let (k, v) = crate::repeat_kv(k, v, repeats)
            .expect("failed to call repeat_kv");

        let ys = if true {
            // use pytorch impl (currently not using a metal kernel on macOS)
            crate::scaled_dot_product_attention(q, k, v)
                .expect("failed to call scaled dot product attention")
        } else {
            // Causaul Self Attention
            let attn = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(head_size as f64));
            let attn = attn
                .masked_fill(&self.mask.i((.., .., ..L, ..L)).eq(0.), f64::NEG_INFINITY)
                .softmax(-1, xs.kind())
                .dropout(self.attn_pdrop, train);
            attn.matmul(&v)
        };

        // reassembly head outputs side by side
        let ys = ys.transpose(1, 2).contiguous().view([B, L, self.n_head * self.head_dim]);

        // output projection
        ys.apply(&self.proj).dropout(self.resid_pdrop, train)
    }
}
