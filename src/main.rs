use tch::{
    data::TextData,
    nn::{self, layer_norm, LayerNorm, Module, ModuleT, OptimizerConfig, Path},
    Device, IndexOp, Kind, Tensor,
};

const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 4096;

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

#[derive(Debug)]
struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    fn new(vs: Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(WEIGHT_DECAY_GROUP);
        let no_wd = vs.set_group(NO_WEIGHT_DECAY_GROUP);
        Self {
            // x{in_dim} * w{out_dim, in_dim} + b{in_dim}
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
            bs: no_wd.zeros("bias", &[out_dim]),
        }
    }

    fn new_no_bias(vs: Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(WEIGHT_DECAY_GROUP);
        let no_wd = vs.set_group(NO_WEIGHT_DECAY_GROUP);
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

#[derive(Debug)]
struct Attention {
    n_head: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    key: Linear,
    query: Linear,
    value: Linear,
    proj: Linear,
    mask: Tensor,
}

impl Attention {
    fn new(p: &Path, cfg: &Config) -> Self {
        let key = Linear::new(p / "key", cfg.n_embd, cfg.n_embd);
        let query = Linear::new(p / "query", cfg.n_embd, cfg.n_embd);
        let value = Linear::new(p / "value", cfg.n_embd, cfg.n_embd);
        let proj = Linear::new(p / "proj", cfg.n_embd, cfg.n_embd);
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

impl ModuleT for Attention {
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
struct Block<Attn: ModuleT> {
    resid_pdrop: f64,
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Attn,
    lin1: Linear,
    lin2: Linear,
}

impl<Attn: ModuleT> Block<Attn> {
    fn new(p: &Path, cfg: &Config, attn: Attn) -> Self {
        let norm1 = layer_norm(p / "ln1", vec![cfg.n_embd], Default::default());
        let norm2 = layer_norm(p / "ln2", vec![cfg.n_embd], Default::default());
        let lin1 = Linear::new(p / "lin1", cfg.n_embd, 4 * cfg.n_embd);
        let lin2 = Linear::new(p / "lin2", 4 * cfg.n_embd, cfg.n_embd);
        Self {
            resid_pdrop: cfg.resid_pdrop,
            norm1,
            norm2,
            attn,
            lin1,
            lin2,
        }
    }
}

impl<Attn: ModuleT> ModuleT for Block<Attn> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply_t(&self.attn, train);
        let ys = xs
            .apply(&self.norm2)
            .apply(&self.lin1)
            .gelu("none")
            .apply(&self.lin2)
            .dropout(self.resid_pdrop, train);
        xs + ys
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
        let head = Linear::new_no_bias(p / "head", cfg.n_embd, cfg.vocab_size);
        let mut blocks = nn::seq_t();
        for i in 0..cfg.n_layer {
            let attn = Attention::new(p, cfg);
            blocks = blocks.add(Block::new(&(p / i), cfg, attn));
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

    let device = Device::Mps;
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
