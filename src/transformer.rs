use tch::{
    nn::{self, layer_norm, Module, ModuleT, Path},
    IndexOp, Kind, Tensor,
};

use crate::linear::Linear;

#[derive(Debug, Clone)]
pub struct Config {
    pub kind: Kind,
    pub vocab_size: i64,
    /// common model dimension. aka "hidden dimension"
    pub dim: i64,
    /// feedforward internal dimension. aka "intermediate dimension"
    pub hidden_dim: i64,
    /// TODO from mistral, what is this?
    pub head_dim: i64,
    /// number of attention heads
    pub n_head: i64,
    /// TODO from mistral, what is this?
    pub n_kv_heads: i64,
    /// number of block layers
    pub n_layer: i64,
    pub block_size: i64,
    pub attn_pdrop: f64,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    /// Does this model use position encoding before the block layers?
    pub position_encoding: bool,
}

pub trait NormLayer: ModuleT + Module {
    fn new(p: &Path, size: i64, kind: Kind) -> Self;
}

pub trait AttentionLayer: ModuleT {
    fn new(p: &Path, cfg: &Config) -> Self;
}

pub trait FeedForward: ModuleT {
    fn new(p: &Path, dim: i64, hidden_dim: i64, kind: Kind) -> Self;
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
            norm1: Form::Norm::new(&(p / "input_layernorm"), cfg.dim, cfg.kind),
            norm2: Form::Norm::new(&(p / "post_attention_layernorm"), cfg.dim, cfg.kind),
            attn: Form::Attn::new(&(p / "self_attn"), cfg),
            ffn: Form::FF::new(&(p / "mlp"), cfg.dim, cfg.hidden_dim, cfg.kind),
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
    pos_emb: Option<Tensor>,
    ln_f: Form::Norm,
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
            cfg.dim,
            Default::default(),
        );
        let pos_emb = if cfg.position_encoding {
            Some(p.zeros("pos_emb", &[1, cfg.block_size, cfg.dim]))
        } else {
            None
        };
        let ln_f = Form::Norm::new(&(p / "norm"), cfg.dim, cfg.kind);
        let head = Linear::new_no_bias(p / "lm_head", cfg.dim, cfg.vocab_size, cfg.kind);
        let mut blocks = nn::seq_t();
        for i in 0..cfg.n_layer {
            blocks = blocks.add(Block::<Form>::new(&(p / "layers" / i), cfg));
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

        // apply position encodings if they exist
        let xs = if let Some(pos_emb) = &self.pos_emb {
            let pos_emb = pos_emb.i((.., ..sz_t, ..));
            tok_emb + pos_emb
        } else {
            tok_emb
        };

        xs.dropout(self.embd_pdrop, train)
            .apply_t(&self.blocks, train)
            .apply(&self.ln_f)
            .apply(&self.head)
    }
}

// adapt torch::nn::LayerNorm to our trait
impl NormLayer for nn::LayerNorm {
    fn new(p: &Path, size: i64, _kind: Kind) -> Self {
        layer_norm(p, vec![size], Default::default())
    }
}
