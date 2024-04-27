use tch::*;

pub fn _precompute_freqs_cis(seq_len: i64, n_embed: i64, n_head: i64) -> Tensor {
    let n_elem = n_embed / n_head;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
        .collect();
    let arange: Vec<_> = (0..seq_len).map(|c| c as f32).collect();
    let theta = Tensor::from_slice(&theta);
    let arange = Tensor::from_slice(&arange);
    let idx_theta = arange.outer(&theta);
    let shape = [1, 1, seq_len, n_elem / 2, 1];
    let idx_theta_cos = idx_theta.cos().reshape(shape);
    let idx_theta_sin = idx_theta.sin().reshape(shape);
    Tensor::cat(&[&idx_theta_cos, &idx_theta_sin], -1)
}

pub fn _apply_rotary_emb(x: &Tensor, freqs_cis: &Tensor) -> Tensor {
    let mut dims = x.size();
    let v = dims.pop().unwrap();
    dims.push(v / 2);
    dims.push(2);
    let x = x.reshape(&dims);
    let re_x = x.slice(-1, 0, 1, 1);
    let im_x = x.slice(-1, 1, 2, 1);
    let re_f = freqs_cis.slice(-1, 0, 1, 1);
    let im_f = freqs_cis.slice(-1, 1, 2, 1);
    let re = &re_x * &re_f - &im_x * &im_f;
    let im = &re_x * &im_f + &im_x * &re_f;
    let rope = Tensor::cat(&[&re, &im], -1);
    // TODO: Add the flatten op.
    let mut dims = rope.size();
    let v1 = dims.pop().unwrap();
    let v2 = dims.pop().unwrap();
    dims.push(v1 * v2);
    rope.reshape(&dims)
}
