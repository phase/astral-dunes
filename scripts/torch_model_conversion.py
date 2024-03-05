# This script is used to convert a PyTorch model to SafeTensors format.
# It also allows renaming of tensors.
# TODO Can this be replaced with Rust calling the safetensors lib?
# Based on https://github.com/LaurentMazare/tch-rs/blob/main/examples/llama/convert_checkpoint.py
# which is based on https://github.com/Lightning-AI/lit-llama/blob/main/scripts/convert_checkpoint.py
import sys
import torch
from typing import Dict
from pathlib import Path
from safetensors.torch import save_file

def rename_tensors(
        state_dict: Dict[str, torch.Tensor],
        weight_name_map: Dict[str, str],
        dtype: torch.dtype = torch.float16
) -> Dict[str, torch.Tensor]:
    model = {}
    for k, v in state_dict.items():
        print(f"{k} {v.shape} {v.dtype}")
        model[weight_name_map.get(k, k)] = v.to(dtype)
    return model

mistral_to_safetensors = {
    #'model.layers.0.self_attn.k_proj.weight': 'model.layers.0.attn.key.weight',
}

def convert_weights(model_ckpt, *, output_path: Path, dtype: str = "float32") -> None:
    dt = getattr(torch, dtype, None)
    if dt is None:
        raise ValueError(f"Invalid dtype: {dtype}")
    print(f"Loading model from {model_ckpt}")
    model = torch.load(model_ckpt, map_location="cpu")
    print("Converting Torch weights to SafeTensors")
    new_model = rename_tensors(model, mistral_to_safetensors, dt)
    del model
    #save_file(new_model, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python torch_model_conversion.py <input_model> <output_model>.safetensors")
        sys.exit(1)
    convert_weights(sys.argv[1], output_path=Path(sys.argv[2]))
