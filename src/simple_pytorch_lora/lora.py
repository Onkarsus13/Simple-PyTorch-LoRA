import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List

@dataclass
class LoRAConfig:
    r: int = 4  # rank of the low-rank decomposition
    alpha: int = 16  # scaling factor
    dropout: float = 0.0  # dropout probability for LoRA
    target_modules: List[str] = None  # substrings of module names to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'conv']

class LoRALayer(nn.Module):
    def __init__(self, orig_module: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.orig_module = orig_module
        self.r = config.r
        self.scaling = config.alpha / config.r
        # Freeze
        orig_module.weight.requires_grad = False
        if orig_module.bias is not None:
            orig_module.bias.requires_grad = False
        # LoRA params
        in_f, out_f = orig_module.in_features, orig_module.out_features
        self.lora_A = nn.Parameter(torch.zeros((self.r, in_f)))
        self.lora_B = nn.Parameter(torch.zeros((out_f, self.r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.orig_module(x)
        lora_out = self.lora_B @ (self.lora_A @ self.dropout(x).T)
        return result + (lora_out.T) * self.scaling

class LoRAConv1d(nn.Module):
    def __init__(self, orig_module: nn.Conv1d, config: LoRAConfig):
        super().__init__()
        self.orig_module = orig_module
        self.r = config.r
        self.scaling = config.alpha / config.r
        # Freeze
        orig_module.weight.requires_grad = False
        if orig_module.bias is not None:
            orig_module.bias.requires_grad = False
        # Adapters
        self.lora_A = nn.Conv1d(orig_module.in_channels, self.r,
                                 kernel_size=orig_module.kernel_size,
                                 stride=orig_module.stride,
                                 padding=orig_module.padding,
                                 dilation=orig_module.dilation,
                                 groups=orig_module.groups,
                                 bias=False)
        self.lora_B = nn.Conv1d(self.r, orig_module.out_channels,
                                 kernel_size=1,
                                 bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.orig_module(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return result + lora_out * self.scaling

class LoRAConv2d(nn.Module):
    def __init__(self, orig_module: nn.Conv2d, config: LoRAConfig):
        super().__init__()
        self.orig_module = orig_module
        self.r = config.r
        self.scaling = config.alpha / config.r
        orig_module.weight.requires_grad = False
        if orig_module.bias is not None:
            orig_module.bias.requires_grad = False
        self.lora_A = nn.Conv2d(orig_module.in_channels, self.r,
                                 kernel_size=orig_module.kernel_size,
                                 stride=orig_module.stride,
                                 padding=orig_module.padding,
                                 dilation=orig_module.dilation,
                                 groups=orig_module.groups,
                                 bias=False)
        self.lora_B = nn.Conv2d(self.r, orig_module.out_channels,
                                 kernel_size=1,
                                 bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.orig_module(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return result + lora_out * self.scaling

class LoRAConv3d(nn.Module):
    def __init__(self, orig_module: nn.Conv3d, config: LoRAConfig):
        super().__init__()
        self.orig_module = orig_module
        self.r = config.r
        self.scaling = config.alpha / config.r
        orig_module.weight.requires_grad = False
        if orig_module.bias is not None:
            orig_module.bias.requires_grad = False
        self.lora_A = nn.Conv3d(orig_module.in_channels, self.r,
                                 kernel_size=orig_module.kernel_size,
                                 stride=orig_module.stride,
                                 padding=orig_module.padding,
                                 dilation=orig_module.dilation,
                                 groups=orig_module.groups,
                                 bias=False)
        self.lora_B = nn.Conv3d(self.r, orig_module.out_channels,
                                 kernel_size=1,
                                 bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.orig_module(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return result + lora_out * self.scaling


def apply_lora(model: nn.Module, config: LoRAConfig):
    for name, module in list(model.named_children()):
        apply_lora(module, config)
        if isinstance(module, nn.Linear) and any(t in name for t in config.target_modules):
            setattr(model, name, LoRALayer(module, config))
        elif isinstance(module, nn.Conv1d) and any('conv1d' in name.lower() or 'conv' in name.lower() for t in config.target_modules):
            setattr(model, name, LoRAConv1d(module, config))
        elif isinstance(module, nn.Conv2d) and any('conv2d' in name.lower() or 'conv' in name.lower() for t in config.target_modules):
            setattr(model, name, LoRAConv2d(module, config))
        elif isinstance(module, nn.Conv3d) and any('conv3d' in name.lower() or 'conv' in name.lower() for t in config.target_modules):
            setattr(model, name, LoRAConv3d(module, config))


def merge_lora(model: nn.Module):
    for name, module in list(model.named_children()):
        if isinstance(module, LoRALayer):
            orig = module.orig_module
            delta = module.lora_B @ module.lora_A
            orig.weight.data += delta * module.scaling
            setattr(model, name, orig)
        elif isinstance(module, LoRAConv1d):
            orig = module.orig_module
            A_w = module.lora_A.weight.data  # (r, in, k)
            B_w = module.lora_B.weight.data.view(orig.out_channels, module.r)  # (out, r)
            delta = torch.einsum('or,rik->oik', B_w, A_w) * module.scaling
            orig.weight.data += delta
            setattr(model, name, orig)
        elif isinstance(module, LoRAConv2d):
            orig = module.orig_module
            A_w = module.lora_A.weight.data  # (r, in, kH, kW)
            B_w = module.lora_B.weight.data.view(orig.out_channels, module.r)  # (out, r)
            delta = torch.einsum('or,rijk->oijk', B_w, A_w) * module.scaling
            orig.weight.data += delta
            setattr(model, name, orig)
        elif isinstance(module, LoRAConv3d):
            orig = module.orig_module
            A_w = module.lora_A.weight.data  # (r, in, kD, kH, kW)
            B_w = module.lora_B.weight.data.view(orig.out_channels, module.r)
            delta = torch.einsum('or,r...->o...', B_w, A_w) * module.scaling
            orig.weight.data += delta
            setattr(model, name, orig)
        else:
            merge_lora(module)

# Example usage
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
    config = LoRAConfig(r=8, alpha=32, dropout=0.1, target_modules=['attn', 'conv'])
    apply_lora(model, config)
    # ... train ...
    merge_lora(model)
    print("Merged all LoRA modules for inference.")
