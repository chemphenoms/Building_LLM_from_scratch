import torch
import torch.nn as nn
import sentencepiece as spm
from torch.utils.checkpoint import checkpoint_sequential

from Models.Llama.common_components import compute_rope, FeedForward, RMSNorm


# ----------------------------- #
# 1. Tokenizer using SentencePiece
# ----------------------------- #
class Llama2Tokenizer:
    """
    A wrapper over SentencePiece for tokenizing and detokenizing text.
    """

    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        # Converts string into list of token IDs
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        # Converts list of token IDs back into string
        return self.tokenizer.decode_ids(ids)


# ----------------------------- #
# 2. Rotary Positional Embedding (RoPE) Precomputation
# ----------------------------- #
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    """
    Precomputes sine and cosine matrices for RoPE encoding.

    Args:
        head_dim: Size of one attention head.
        theta_base: Controls frequency decay across dimensions.
        context_length: Maximum sequence length.

    Returns:
        cos, sin matrices for position encoding.
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]  # outer product

    # Duplicate to match head_dim
    angles = torch.cat([angles, angles], dim=1)

    return torch.cos(angles), torch.sin(angles)


# ----------------------------- #
# 3. Multi-Head Self Attention (with RoPE and Causal Masking)
# ----------------------------- #
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention using rotary position encoding and masking for autoregressive generation.
    """

    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear projections for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)

        # Final projection after attention
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Causal mask (upper triangle) to prevent attending to future tokens
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Precompute RoPE sine and cosine matrices
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Apply projections to compute Q, K, V
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional encoding
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(-2, -1) / self.head_dim ** 0.5

        # Apply causal mask (truncated to current sequence length)
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Combine attention weights with values
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)

        return self.out_proj(context)


# ----------------------------- #
# 4. Transformer Block
# ----------------------------- #
class TransformerBlock(nn.Module):
    """
    A transformer block with RMSNorm, attention, and feed-forward submodules.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x):
        # Attention block with residual connection
        residual = x
        x = self.att(self.norm1(x))
        x = x + residual

        # Feedforward block with residual connection
        residual = x
        x = self.ff(self.norm2(x))
        x = x + residual

        return x


# ----------------------------- #
# 5. Full LLaMA2 Model
# ----------------------------- #
class Llama2Model(nn.Module):
    """
    Full LLaMA2-style model: embedding -> transformer layers -> final norm -> output projection.
    """

    def __init__(self, cfg):
        super().__init__()

        # Embedding for input tokens
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # Stack of transformer layers
        layers = [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        # Enable checkpointing for memory efficiency if needed
        self.trf_blocks = nn.Sequential(*layers)

        # Final layer normalization
        self.final_norm = RMSNorm(cfg["emb_dim"])

        # Output projection layer (tied to vocab size)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        """
        Args:
            in_idx (Tensor): Token indices [batch_size, seq_len]

        Returns:
            logits (Tensor): Vocabulary logits [batch_size, seq_len, vocab_size]
        """
        tok_embeds = self.tok_emb(in_idx)
        x = self.trf_blocks(tok_embeds)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
