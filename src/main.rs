use std::collections::HashMap;

use pyo3_tch::PyTensor;
use tch::{
    data::TextData,
    nn::{self, LayerNorm, ModuleT, OptimizerConfig, VarStore},
    Device, IndexOp, Kind, Tensor,
};
use pyo3::{prelude::*, types::{PyDict, PyIterator, PyTuple}};
use safetensors::Dtype;

mod gelu;
mod kernels;
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
const SAMPLING_LEN: i64 = 512;

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

/// TODO: try_into() trait resolution fails with
/// 'for that trait implementation, expected `safetensors::tensor::Dtype`, found `Dtype`'
/// yes, those are indeed the same type...
/// impls: https://github.com/LaurentMazare/tch-rs/blob/4cdee63c3d56ed5663e256e95492796f1125d751/src/tensor/safetensors.rs#L13
fn _convert_dtype_to_kind(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        dtype => panic!("unsupported dtype {dtype:?}"),
    }
}

fn _load_safetensors(file_name: impl AsRef<str>, vs: &VarStore) -> anyhow::Result<()> {
    let file = std::fs::File::open(file_name.as_ref())?;
    let content = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = safetensors::SafeTensors::deserialize(&content)?;

    let mut variables = vs.variables_.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        let view = safetensors.tensor(name)?;
        let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
        let kind: Kind = _convert_dtype_to_kind(view.dtype());
        // Using from_blob here instead of from_data_size avoids unnecessary copies
        let src_tensor = unsafe {
            Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu)
        };
        var.f_copy_(&src_tensor)?;
    }
    Ok(())
}

fn load_torch_model(model_location: String) -> PyResult<Vec<(String, Tensor)>> {
    Python::with_gil(|py| {
        // TODO cache loaded modules
        let fun: PyObject = PyModule::from_code(
            py,
            include_str!("load_model.py"),
            "load_model.py",
            "load_model",
        )?
        .getattr("load_model")?
        .into();

        // call the function & get the state_dict
        let dict: PyObject = fun.call1(py,  (model_location,))?;
        let dict: Py<PyDict> = dict.downcast::<PyDict>(py)?.into();
        // get the items() out
        // this is an iterator of (String, Tensor)
        let items: PyObject = dict.call_method0(py, "items")?;
        let items: &PyIterator = items.as_ref(py).iter()?;

        let mut model_vars = Vec::new();
        for item in items {
            let item = item?.downcast::<PyTuple>()?;
            if let &[name, tensor] = item.as_slice() {
                let name = name.extract::<String>()?;
                let tensor = tensor.extract::<PyTensor>()?.0;
                model_vars.push((name, tensor))
            }
        }

        Ok(model_vars)
    })
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
        head_dim: 128,
        hidden_dim: 14336,
        n_head: 32,
        n_kv_heads: 8,
        n_layer: 32,
        block_size: 16,//1024,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
        position_encoding: false,
    };

    let cfg = Config {
        kind: Kind::BFloat16,
        vocab_size: labels,
        dim: 512,
        head_dim: 64,
        hidden_dim: 512*4,
        n_head: 16,
        n_kv_heads: 4,
        n_layer: 16,
        block_size: 128,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
        position_encoding: false,
    };

    println!("Building model");
    let model = Transformer::<LlamaConfig>::new(&(vs.root() / "model"), &cfg);

    let model_size = vs.trainable_variables().iter().fold(0, |sum, tensor| {
        sum + tensor.size().iter().fold(1, |a, b| a * b)
    });
    println!("Model size: {}. Context window: {}", model_size, cfg.block_size);
    crate::mem::debug_memory("Memory usage before training");

    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("Usage: astral-dunes (train|predict weights.ot seqstart)")
    }

    match args[1].as_str() {
        "inspect" => {
            let torch_model = args[2].as_str().to_string();
            let source_vars = load_torch_model(torch_model)?;
            let source_map: HashMap<_, _> = source_vars.into_iter().collect();

            let mut expected_vars = Vec::with_capacity(vs.variables().len());
            for (name, tensor) in vs.variables().iter() {
                expected_vars.push((name.clone(), tensor.size()));
            }

            expected_vars.sort_by(|a, b| a.0.cmp(&b.0));

            for (name, tensor) in expected_vars.iter() {
                if let Some(source_tensor) = source_map.get(name) {
                    if *tensor != *source_tensor.size() {
                        println!("{}: expected {:?} but got {:?}", name, tensor, source_tensor.size());
                    } else {
                        println!("{}: ok", name);
                    }
                } else {
                    println!("{}: missing", name);
                }
            }
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
            println!("{}", sample(&data, &model, input, &cfg));
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
                    let logits = xs.apply_t(&model, true);
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

                        let output: String = sample(&data, &model, input, &cfg);
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

/// TODO dont reparse module every call lol
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
