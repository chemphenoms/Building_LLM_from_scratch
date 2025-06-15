import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

# ---------------------- Multi-Head Attention ---------------------- #
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer as used in transformer models.
    Think of it as letting each word attend to other words to decide what matters.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Output dimension must be divisible by number of heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Each head handles a part of the total features

        # Create 3 different projections for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Final projection to combine heads
        self.dropout = nn.Dropout(dropout)

        # A triangular mask to ensure the model only attends to previous tokens (causal)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape  # Batch size, sequence length, feature dim

        # Project input into Q, K, V spaces
        Q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = Q @ K.transpose(2, 3) / self.head_dim ** 0.5
        mask = self.mask[:num_tokens, :num_tokens].to(attn_scores.device).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = attn_weights @ V
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)

        return self.out_proj(context)

# ---------------------- Feed Forward Block ---------------------- #
class FeedForward(nn.Module):
    """
    A simple MLP block used after attention to transform each token separately.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),  # Non-linear activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------- Transformer Block ---------------------- #
class TransformerBlock(nn.Module):
    """
    One complete transformer block: Attention -> Add & Norm -> FeedForward -> Add & Norm
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.dropout(self.att(self.norm1(x)))
        # FeedForward block with residual connection
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# ---------------------- Full GPT Model ---------------------- #
class GPTModel(nn.Module):
    """
    The full GPT-style model: token + positional embedding → stack of transformer blocks → final logits.
    """
    def __init__(self, cfg, use_actv_ckpt=False):
        super().__init__()
        self.cfg = cfg
        self.use_actv_ckpt = use_actv_ckpt

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # Learn word meanings
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # Learn position order
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = nn.LayerNorm(cfg["emb_dim"])
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        B, T = x.shape  # Batch size, sequence length
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # Position indices

        x = self.tok_emb(x) + self.pos_emb(pos)  # Combine token and position
        x = self.drop_emb(x)

        if self.use_actv_ckpt:
            x = checkpoint_sequential(self.blocks, segments=self.cfg["n_layers"], input=x, use_reentrant=False)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        logits = self.output_head(x)

        print(f"Shape Debug → Input: {x.shape}, Output Logits: {logits.shape}")
        return logits

# ---------------------- Configuration ---------------------- #
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "eos_id": 50256,
    "eos_text": "<|endoftext|>"
}

# ---------------------- Synthetic Test ---------------------- #
def test_gpt_model():
    print("\nRunning synthetic test for GPT model...")
    model = GPTModel(GPT_CONFIG_SMALL)
    dummy_input = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (2, GPT_CONFIG_SMALL["context_length"]))
    output = model(dummy_input)
    assert output.shape == (2, GPT_CONFIG_SMALL["context_length"], GPT_CONFIG_SMALL["vocab_size"])
    print("✅ Test passed: Output shape is correct!")

if __name__ == "__main__":
    test_gpt_model()
