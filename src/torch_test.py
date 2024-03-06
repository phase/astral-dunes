import torch
from torch import nn

def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values

def scaled_dot_product_attention(q, k, v):
    return nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.1,
        is_causal=True,
    )

def torch_example():
    x = torch.rand(2, 2)
    print(type(x))
    print(x)
    return x
