use pyo3_tch::PyTensor;
use tch::{
    data::TextData,
    nn::{self, LayerNorm, ModuleT, OptimizerConfig},
    Device, IndexOp, Kind, Tensor
};
use pyo3::prelude::*;

mod gelu;
mod linear;
mod mem;
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
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 2048;

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

fn sample(data: &TextData, model: &impl ModuleT, input: Tensor, cfg: &Config) -> String {
    let mut input = input;
    let mut result = String::new();

    for _ in 0..SAMPLING_LEN {
        let logits = input.apply_t(model, false).i((0, -1, ..));
        let sampled_y = logits.softmax(-1, cfg.kind).multinomial(1, true);
        let last_label = i64::try_from(&sampled_y).unwrap();
        result.push(data.label_to_char(last_label));
        input = Tensor::cat(&[input, sampled_y.view([1, 1])], 1).narrow(1, 1, cfg.block_size);
    }

    result
}

fn main() -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python();

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

    let _mistral = Config {
        kind: Kind::BFloat16,
        vocab_size: labels,
        dim: 4096,
        hidden_dim: 14336,
        n_head: 8,
        n_layer: 32,
        block_size: 1024,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
        head_dim: 128,
        n_kv_heads: 8,
    };

    let cfg = Config {
        kind: Kind::BFloat16,
        vocab_size: labels,
        dim: 512,
        hidden_dim: 512 * 4,
        n_head: 8,
        n_layer: 8,
        block_size: 128,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
        head_dim: 128,
        n_kv_heads: 8,
    };

    println!("Building model");
    let gpt = Transformer::<LlamaConfig>::new(&vs.root(), &cfg);

    let model_size = vs.trainable_variables().iter().fold(0, |sum, tensor| {
        sum + tensor.size().iter().fold(1, |a, b| a * b)
    });
    println!("Model size: {}. Context window: {}", model_size, cfg.block_size);

    // print memory usage
    crate::mem::debug_memory("Memory usage before training");

    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("Usage: astral-dunes (train|predict weights.ot seqstart)")
    }

    match args[1].as_str() {
        "inspect" => {
            test_loading_pytorch()?;
        },
        "predict" => {
            // load varstore
            vs.load(args[3].as_str())?;

            let start = args[3].as_str();
            let input = Tensor::zeros([1, cfg.block_size], (Kind::Int64, device));
            for (idx, c) in start.chars().rev().enumerate() {
                let idx = idx as i64;
                if idx >= cfg.block_size {
                    break;
                }
                let label = data.char_to_label(c)? as i64;
                let _ = input.i((0, cfg.block_size - 1 - idx)).fill_(label);
            }
            println!("{}", sample(&data, &gpt, input, &cfg));
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
                for batch in data.iter_shuffle(cfg.block_size + 1, BATCH_SIZE) {
                    let xs = batch
                        .narrow(1, 0, cfg.block_size)
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    let ys = batch
                        .narrow(1, 1, cfg.block_size)
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    let logits = xs.apply_t(&gpt, true);
                    let loss = logits
                        .view([BATCH_SIZE * cfg.block_size, labels])
                        .cross_entropy_for_logits(&ys.view([BATCH_SIZE * cfg.block_size]));
                    opt.backward_step_clip(&loss, 0.5);
                    sum_loss += f64::try_from(&loss)?;
                    cnt_loss += 1.;
                    idx += 1;

                    let perplexity = 2_f64.powf(sum_loss / cnt_loss);
                    crate::mem::debug_memory(format!("epoch={:4} batch={:4} ppl={:5.3} mem", epoch, idx, perplexity));

                    if idx % 500 == 0 {
                        println!("epoch={:4} loss={:5.3}", epoch, sum_loss / cnt_loss);

                        println!(".. testing inference");
                        let input = Tensor::zeros([1, cfg.block_size], (Kind::Int64, device));

                        let output: String = sample(&data, &gpt, input, &cfg);
                        // save output & weights to disk
                        println!(".. saving weights");
                        let filename = format!("data/{experiment}/model{idx}.txt");
                        println!("{}", output);
                        std::fs::write(filename, output)?;

                        if let Err(err) = vs.save(format!("data/{experiment}/model{idx}.ot")) {
                            println!("error saving model: {err}");
                        }
                        sum_loss = 0.;
                        cnt_loss = 0.;
                    }
                }
            }
        }
        _ => anyhow::bail!("usage: astral-dunes (train|predict weights.ot seqstart)"),
    };
    Ok(())
}

fn test_loading_pytorch() -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            include_str!("torch_test.py"),
            "torch_test.py",
            "torch_test",
        )?
        .getattr("torch_example")?
        .into();

        // call object without any arguments
        let result = fun.call0(py)?.extract::<PyTensor>(py)?.0;
        println!("size: {:?}", result.size());

        let result = scaled_dot_product_attention(result.copy(), result.copy(), result).unwrap();
        println!("size: {:?}", result.size());

        Ok(())
    })
}

pub fn scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor) -> PyResult<Tensor> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            include_str!("torch_test.py"),
            "torch_test.py",
            "torch_test",
        )?
        .getattr("scaled_dot_product_attention")?
        .into();

        // call object without any arguments
        let result = fun.call1(py, (PyTensor(q), PyTensor(k), PyTensor(v)))?.extract::<PyTensor>(py)?.0;
        Ok(result)
    })
}

pub fn repeat_kv(k: Tensor, v: Tensor, repeats: i64) -> PyResult<(Tensor, Tensor)> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            include_str!("torch_test.py"),
            "torch_test.py",
            "torch_test",
        )?
        .getattr("repeat_kv")?
        .into();

        // call object without any arguments
        let result = fun.call1(py, (PyTensor(k), PyTensor(v), repeats, 1))?.extract::<(PyTensor, PyTensor)>(py)?;
        Ok((result.0.0, result.1.0))
    })
}
