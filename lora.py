import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    Implements a single LoRA layer: W + (alpha / rank) * (A @ B)
    where A and B are low-rank matrices added to the original weight.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float, dtype=torch.float32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        # A: projects input to low-rank
        self.A = nn.Parameter(torch.empty(in_dim, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # B: projects back to output space
        self.B = nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply LoRA adaptation with scaling factor
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """
    Wraps a standard Linear layer with LoRA adaptation.
    During forward pass: output = original_linear(x) + lora(x)
    """
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dtype=torch.float32):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float, dtype=torch.float32):
    """
    Recursively traverses a model and replaces all nn.Linear layers with LinearWithLoRA.

    Args:
        model (nn.Module): The PyTorch model to modify.
        rank (int): LoRA rank (dimension of low-rank adaptation).
        alpha (float): LoRA scaling parameter.
        dtype (torch.dtype): Data type to use for LoRA parameters.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear with LoRA-enhanced Linear
            setattr(model, name, LinearWithLoRA(module, rank, alpha, dtype))
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(module, rank, alpha, dtype)
