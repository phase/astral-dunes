use tch::{
    data::TextData,
    nn::{self, LayerNorm, ModuleT, OptimizerConfig},
    Device, IndexOp, Kind, Tensor
};

mod gelu;
mod linear;
mod rmsnorm;
mod rope;
mod self_attention;
mod swish;
mod transformer;

use gelu::Gelu;
use rmsnorm::RmsNorm;
use swish::Swish;
use self_attention::SelfAttention;
use transformer::*;

pub const NO_WEIGHT_DECAY_GROUP: usize = 0;
pub const WEIGHT_DECAY_GROUP: usize = 1;
const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 4096;

#[derive(Debug)]
pub struct GPTConfig;

impl BlockConfig for GPTConfig {
    type Attn = SelfAttention;
    type Norm = LayerNorm;
    type FF = Gelu;
}

#[derive(Debug)]
pub struct LlamaConfig;

impl BlockConfig for LlamaConfig {
    type Attn = SelfAttention; //<RopeEmbedding>;
    type Norm = RmsNorm;
    type FF = Swish;
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
    let experiment = "zig-compiler";
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

    let gpt = Transformer::<GPTConfig>::new(&(vs.root() / "gpt"), &cfg);
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

                    if idx % 10 == 0 && idx % 500 != 0  {
                        println!("epoch: {:4} | batch: {:4}", epoch, idx);
                    } else if idx == 5 || idx % 500 == 0 {
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
