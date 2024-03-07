import torch

# Included & called by main.rs
def load_model(model_checkpoint):
    return torch.load(model_checkpoint, map_location="cpu")
