use tch::{
    data::TextData,
    nn::{self, layer_norm, LayerNorm, Module, ModuleT, OptimizerConfig, Path, Linear, linear},
    Device, IndexOp, Kind, Tensor
};

const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 4096;

const NO_BIAS: nn::LinearConfig = nn::LinearConfig { ws_init: nn::init::DEFAULT_KAIMING_UNIFORM, bs_init: None, bias: false, };

#[derive(Debug, Clone)]
struct Config {
    vocab_size: i64,
    n_embd: i64,
    n_head: i64,
    n_layer: i64,
    block_size: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    embd_pdrop: f64,
}

const NO_WEIGHT_DECAY_GROUP: usize = 0;
const WEIGHT_DECAY_GROUP: usize = 1;


/// Root Mean Square Layer Normalization
#[derive(Debug)]
struct RmsNorm {
    scale: Tensor,
    size: i64,
}

impl RmsNorm {
    fn new(p: &Path, size: i64) -> Self {
        let scale = p.zeros("scale", &[size]);
        Self { scale, size }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let norm = (xs*xs).mean_dim(-1, true, Kind::Float);
        let xs_norm = xs * (norm + 1e-5).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * xs_norm
    }
}

trait Norm: ModuleT + Module {
    fn new(p: &Path, size: i64) -> Self;
}

impl Norm for LayerNorm {
    fn new(p: &Path, size: i64) -> Self {
        layer_norm(p, vec![size], Default::default())
    }
}

impl Norm for RmsNorm {
    fn new(p: &Path, size: i64) -> Self {
        RmsNorm::new(p, size)
    }
}

trait Attention: ModuleT {
    fn new(p: &Path, cfg: &Config) -> Self;
}

/// Transformer Decoder-only Self Attention
#[derive(Debug)]
struct SelfAttention {
    n_head: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    key: Linear,
    query: Linear,
    value: Linear,
    proj: Linear,
    mask: Tensor,
}

impl Attention for SelfAttention {
    fn new(p: &Path, cfg: &Config) -> Self {
        let key = linear(p / "key", cfg.n_embd, cfg.n_embd, NO_BIAS);
        let query = linear(p / "query", cfg.n_embd, cfg.n_embd, NO_BIAS);
        let value = linear(p / "value", cfg.n_embd, cfg.n_embd, NO_BIAS);
        let proj = linear(p / "proj", cfg.n_embd, cfg.n_embd, NO_BIAS);
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

#[derive(Debug)]
struct Block<A: Attention, N: Norm, M: MLP> {
    resid_pdrop: f64,
    norm1: N,
    norm2: N,
    attn: A,
    mlp: M,
}

impl<A: Attention, N: Norm, M: MLP> Block<A, N, M> {
    fn new(p: &Path, cfg: &Config) -> Self {
        Self {
            resid_pdrop: cfg.resid_pdrop,
            norm1: N::new(&(p / "ln1"), cfg.n_embd),
            norm2: N::new(&(p / "ln2"), cfg.n_embd),
            attn: A::new(&(p / "attn"), cfg),
            mlp: M::new(&(p / "mlp"), cfg.n_embd, 4 * cfg.n_embd, cfg.n_embd),
        }
    }
}

impl<A: Attention, N: Norm, M: MLP> ModuleT for  Block<A, N, M> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply_t(&self.attn, train);
        let ys = xs
            .apply(&self.norm2)
            .apply_t(&self.mlp, train)
            .dropout(self.resid_pdrop, train);
        xs + ys
    }
}

#[derive(Debug)]
struct GeluMLP {
    lin1: Linear,
    lin2: Linear,
}

impl Module for GeluMLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.lin1).gelu("none").apply(&self.lin2)
    }
}

impl MLP for GeluMLP {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
        let lin1 = linear(p / "lin1", in_dim, hidden_dim, NO_BIAS);
        let lin2 = linear(p / "lin2", hidden_dim, out_dim, NO_BIAS);
        Self { lin1, lin2 }
    }
}

trait MLP: ModuleT {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self;
}


/// Multi-layer Perceptron using the Sigmoid Linear Unit (aka Swish) activation function
#[derive(Debug)]
struct SwishMLP {
    lin1: Linear,
    lin2: Linear,
}

impl Module for SwishMLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // silu(x) = x * sigmoid(x)
        xs.apply(&self.lin1).silu().apply(&self.lin2)
    }
}

impl MLP for SwishMLP {
    fn new(p: &Path, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
        let lin1 = linear(p / "lin1", in_dim, hidden_dim, NO_BIAS);
        let lin2 = linear(p / "lin2", hidden_dim, out_dim, NO_BIAS);
        Self { lin1, lin2 }
    }
}

#[derive(Debug)]
struct GPT {
    embd_pdrop: f64,
    tok_emb: nn::Embedding,
    pos_emb: Tensor,
    ln_f: LayerNorm,
    head: Linear,
    blocks: nn::SequentialT,
}

impl GPT {
    fn new(p: &Path, cfg: &Config) -> Self {
        let tok_emb = nn::embedding(
            p / "tok_emb",
            cfg.vocab_size,
            cfg.n_embd,
            Default::default(),
        );
        let pos_emb = p.zeros("pos_emb", &[1, cfg.block_size, cfg.n_embd]);
        let ln_f = layer_norm(p / "ln_f", vec![cfg.n_embd], Default::default());
        let head = linear(p / "head", cfg.n_embd, cfg.vocab_size, NO_BIAS);
        let mut blocks = nn::seq_t();
        for i in 0..cfg.n_layer {
            blocks = blocks.add(Block::<SelfAttention, RmsNorm, SwishMLP>::new(&(p / i), cfg));
        }
        Self {
            embd_pdrop: cfg.embd_pdrop,
            tok_emb,
            pos_emb,
            ln_f,
            head,
            blocks,
        }
    }
}

impl ModuleT for GPT {
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

fn sample(data: &TextData, model: &impl ModuleT, input: Tensor) -> String {
    let mut input = input;
    let mut result = String::new();

    for _ in 0..SAMPLING_LEN {
        let logits = input.apply_t(model, false).i((0, -1, ..));
        let sampled_y = logits.softmax(-1, Kind::Float).multinomial(1, true);
        let last_label = i64::try_from(&sampled_y).unwrap();
        result.push(data.label_to_char(last_label));
        input = Tensor::cat(&[input, sampled_y.view([1, 1])], 1).narrow(1, 1, BLOCK_SIZE);
    }

    result
}

fn main() -> anyhow::Result<()> {
    let experiment = "shakespeare";
    std::fs::create_dir_all(format!("data/{experiment}"))?;

    let device = if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::cuda_if_available()
    };
    let mut vs = nn::VarStore::new(device);

    let data = TextData::new(format!("data/{experiment}.txt"))?;

    let labels = data.labels();
    println!("labels: {:?}", labels);

    let cfg = Config {
        vocab_size: labels,
        n_embd: 512,
        n_head: 8,
        n_layer: 8,
        block_size: BLOCK_SIZE,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
    };

    let gpt = GPT::new(&(vs.root() / "gpt"), &cfg);
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: main (train|predict weights.ot seqstart)")
    }

    match args[1].as_str() {
        "predict" => {
            // load varstore
            vs.load(args[3].as_str())?;

            let start = args[3].as_str();
            let input = Tensor::zeros([1, BLOCK_SIZE], (Kind::Int64, device));
            for (idx, c) in start.chars().rev().enumerate() {
                let idx = idx as i64;
                if idx >= BLOCK_SIZE {
                    break;
                }
                let label = data.char_to_label(c)? as i64;
                let _ = input.i((0, BLOCK_SIZE - 1 - idx)).fill_(label);
            }
            println!("{}", sample(&data, &gpt, input));
        }
        "train" => {
            println!("Starting Training");
            let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE)?;
            opt.set_weight_decay_group(NO_WEIGHT_DECAY_GROUP, 0.0);
            opt.set_weight_decay_group(WEIGHT_DECAY_GROUP, 0.1);
            let mut idx = 0;
            for epoch in 1..(1 + EPOCHS) {
                let mut sum_loss = 0.;
                let mut cnt_loss = 0.;
                for batch in data.iter_shuffle(BLOCK_SIZE + 1, BATCH_SIZE) {
                    let xs = batch
                        .narrow(1, 0, BLOCK_SIZE)
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    let ys = batch
                        .narrow(1, 1, BLOCK_SIZE)
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    let logits = xs.apply_t(&gpt, true);
                    let loss = logits
                        .view([BATCH_SIZE * BLOCK_SIZE, labels])
                        .cross_entropy_for_logits(&ys.view([BATCH_SIZE * BLOCK_SIZE]));
                    opt.backward_step_clip(&loss, 0.5);
                    sum_loss += f64::try_from(&loss)?;
                    cnt_loss += 1.;
                    idx += 1;

                    if idx % 10 == 0 && idx % 200 != 0  {
                        println!("epoch: {:4} | batch: {:4}", epoch, idx);
                    } else if idx == 5 || idx % 200 == 0 {
                        println!("epoch: {:4} | loss: {:5.3}", epoch, sum_loss / cnt_loss);

                        println!(".. testing inference");
                        let input = Tensor::zeros([1, BLOCK_SIZE], (Kind::Int64, device));
                        let output: String = sample(&data, &gpt, input);

                        // save output & weights to disk
                        println!(".. saving weights");
                        let filename = format!("data/{experiment}/gpt{idx}.txt");
                        println!("{}", output);
                        std::fs::write(filename, output)?;

                        if let Err(err) = vs.save(format!("data/{experiment}/gpt{idx}.ot")) {
                            println!("error saving model: {err}");
                        }
                        sum_loss = 0.;
                        cnt_loss = 0.;
                    }
                }
            }
        }
        _ => anyhow::bail!("usage: main (train|predict weights.ot seqstart)"),
    };
    Ok(())
}
