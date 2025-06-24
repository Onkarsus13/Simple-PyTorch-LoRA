import torch
import torch.nn as nn
from simple_pytorch_lora import LoRAConfig, apply_lora, merge_lora

def test_linear_lora_forward():
    linear = nn.Linear(10, 8)
    cfg = LoRAConfig(r=2, alpha=4)
    apply_lora(nn.Sequential(linear), cfg)
    x = torch.randn(3, 10)
    _ = linear(x)

# Add more tests for conv modules and merging
