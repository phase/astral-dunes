import torch
from torch import nn

def scaled_dot_product_attention(q, k, v):
    return nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0,
        is_causal=True,
    )

def torch_example():
    x = torch.rand(2, 2)
    print(type(x))
    print(x)
    return x
