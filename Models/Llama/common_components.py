import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope(x, cos, sin):
    """
    Applies Rotary Positional Embedding (RoPE) to the input tensor.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos (Tensor): Cosine values for position encoding, shape (seq_len, head_dim)
        sin (Tensor): Sine values for position encoding, shape (seq_len, head_dim)

    Returns:
        Tensor: The input with RoPE applied.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape

    # RoPE works only when head dimension is even, as it is split in half
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE to apply."

    # Split the head dimension into two halves: x1 and x2
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Expand sin and cos to match input shape for broadcasting
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply RoPE rotation: combines original tensor and rotated version
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    Rescales the frequency tensor used in RoPE when the sequence length changes.

    Args:
        theta_old (Tensor): Original theta tensor (frequency components).
        context_length_old (int): Original sequence length.
        context_length_new (int): New sequence length to adapt to.

    Returns:
        Tensor: Rescaled theta tensor.
    """
    scaling_factor = context_length_new / context_length_old
    return theta_old * scaling_factor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A normalization strategy that uses RMS instead of mean and variance like LayerNorm.
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim).float())  # Learnable scale parameter

    def forward(self, x):
        # Compute the mean of squared elements for normalization
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)  # Normalize with RMS
        return (x_normed * self.weight).to(dtype=x.dtype)


# ✅ The above RMSNorm can be replaced for robustness and better integration with:
# torch.nn.RMSNorm if using PyTorch >= 2.1
# Example: self.norm = nn.RMSNorm(emb_dim, eps=1e-5)


class SiLU(nn.Module):
    """
    Implements the SiLU (Sigmoid Linear Unit) activation function.
    
    SiLU(x) = x * sigmoid(x), also known as the swish activation.
    """
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# ✅ The above SiLU can be replaced with a built-in version:
# self.silu = nn.SiLU() or use F.silu(x) from torch.nn.functional


class FeedForward(nn.Module):
    """
    FeedForward block used in transformer models.
    
    It applies:
        - Two parallel linear layers to project the input into a higher-dimensional space.
        - Gated activation using SiLU(x1) * x2.
        - A final projection back to the original embedding dimension.

    Args:
        cfg (dict): Configuration dictionary with keys:
            - emb_dim (int): Embedding dimension
            - hidden_dim (int): Hidden layer dimension
            - dtype (torch.dtype): Precision type
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SiLU()

        # ✅ Alternative for cleaner code:
        # self.silu = nn.SiLU() if using torch >= 1.7

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2  # Gated activation mechanism
        return self.fc3(x)
