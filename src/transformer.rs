use tch::{
    nn::{self, layer_norm, LayerNorm, Module, ModuleT, Path},
    IndexOp, Tensor
};

use crate::linear::Linear;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: i64,
    /// common model dimension. aka "hidden dimension"
    pub n_embd: i64,
    /// feedforward internal dimension. aka "intermediate dimension"
    pub ff_int_dim: i64,
    /// number of attention heads
    pub n_head: i64,
    // number of block layers
    pub n_layer: i64,
    pub block_size: i64,
    pub attn_pdrop: f64,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
}

pub trait NormLayer: ModuleT + Module {
    fn new(p: &Path, size: i64) -> Self;
}

pub trait AttentionLayer: ModuleT {
    fn new(p: &Path, cfg: &Config) -> Self;
}

pub trait FeedForward: ModuleT {
    fn new(p: &Path, dim: i64, hidden_dim: i64) -> Self;
}

pub trait BlockConfig: std::fmt::Debug + 'static {
    type Attn: AttentionLayer;
    type Norm: NormLayer;
    type FF: FeedForward;
}

#[derive(Debug)]
pub struct Block<Form: BlockConfig> {
    resid_pdrop: f64,
    norm1: Form::Norm,
    norm2: Form::Norm,
    attn: Form::Attn,
    ffn: Form::FF,
}

impl<Form: BlockConfig> Block<Form> {
    fn new(p: &Path, cfg: &Config) -> Self {
        Self {
            resid_pdrop: cfg.resid_pdrop,
            norm1: Form::Norm::new(&(p / "input_layernorm"), cfg.n_embd),
            norm2: Form::Norm::new(&(p / "post_attention_layernorm"), cfg.n_embd),
            attn: Form::Attn::new(&(p / "self_attn"), cfg),
            ffn: Form::FF::new(&(p / "mlp"), cfg.n_embd, cfg.ff_int_dim),
        }
    }
}

impl<Form: BlockConfig> ModuleT for Block<Form> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply_t(&self.attn, train);
        let ys = xs
            .apply(&self.norm2)
            .apply_t(&self.ffn, train)
            .dropout(self.resid_pdrop, train);
        xs + ys
    }
}

#[derive(Debug)]
pub struct Transformer<Form: BlockConfig> {
    embd_pdrop: f64,
    tok_emb: nn::Embedding,
    pos_emb: Tensor,
    ln_f: LayerNorm,
    head: Linear,
    blocks: nn::SequentialT,
    // fn() keeps this struct Send + Sync
    _form: std::marker::PhantomData<fn(Form) -> Form>,
}

impl<Form: BlockConfig> Transformer<Form> {
    pub fn new(p: &Path, cfg: &Config) -> Self {
        let tok_emb = nn::embedding(
            p / "embed_tokens",
            cfg.vocab_size,
            cfg.n_embd,
            Default::default(),
        );
        let pos_emb = p.zeros("pos_emb", &[1, cfg.block_size, cfg.n_embd]);
        let ln_f = layer_norm(p / "norm", vec![cfg.n_embd], Default::default());
        let head = Linear::new(p / "lm_head", cfg.n_embd, cfg.vocab_size);
        let mut blocks = nn::seq_t();
        for i in 0..cfg.n_layer {
            blocks = blocks.add(Block::<Form>::new(&(p / "model" / "layers" / i), cfg));
        }
        Self {
            embd_pdrop: cfg.embd_pdrop,
            tok_emb,
            pos_emb,
            ln_f,
            head,
            blocks,
            _form: std::marker::PhantomData,
        }
    }
}

impl<Form: BlockConfig> ModuleT for Transformer<Form> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (_sz_b, sz_t) = xs.size2().unwrap();
        let tok_emb = xs.apply(&self.tok_emb);
        let pos_emb = self.pos_emb.i((.., ..sz_t, ..));
        let xs = tok_emb + pos_emb;
        xs.dropout(self.embd_pdrop, train)
            .apply_t(&self.blocks, train)
            .apply(&self.ln_f)
            .apply(&self.head)
    }
}

// adapt torch::nn::LayerNorm to our trait
impl NormLayer for nn::LayerNorm {
    fn new(p: &Path, size: i64) -> Self {
        layer_norm(p, vec![size], Default::default())
    }
}
