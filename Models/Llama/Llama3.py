import torch
import torch.nn as nn
import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from torch.utils.checkpoint import checkpoint_sequential

# Import helper modules (assumed to be implemented)
from Models.Llama.common_components import compute_rope, FeedForward, RMSNorm


# --------------------------- Tokenizer for LLaMA-3 --------------------------------
class Llama3Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        # Define special tokens and reserved tokens
        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if (128002 + i) not in self.special_tokens.values()
        })

        # Create tokenizer object
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


# --------------------------- Shared buffers for RoPE ------------------------------
class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


# --------------------------- Rotary Positional Encoding ---------------------------
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Frequency smoothing config (optional)
    if freq_config:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]
        wavelen = 2 * torch.pi / inv_freq

        inv_freq_adjusted = torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / freq_config["factor"],
            inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed, inv_freq_adjusted)

    positions = torch.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


# --------------------------- Grouped Query Attention ------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, num_kv_groups, rope_base=10_000, rope_config=None, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # Linear projections without bias
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Project input to Q, K, V
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding
        queries = compute_rope(queries, self.cos, self.sin)
        keys = compute_rope(keys, self.cos, self.sin)

        # Expand KV to match query heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention score calculation with mask
        attn_scores = (queries @ keys.transpose(2, 3)) / self.head_dim**0.5
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)

        return self.out_proj(context)


# --------------------------- Transformer Block ------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.att(self.norm1(x).to(torch.bfloat16))
        # Apply feed-forward with residual connection
        x = x + self.ff(self.norm2(x).to(torch.bfloat16))
        return x


# --------------------------- LLaMA-3 Style Model ----------------------------------
class Llama3Model(nn.Module):
    def __init__(self, cfg, use_actv_ckpt=False):
        super().__init__()
        self.cfg = cfg
        self.use_actv_ckpt = use_actv_ckpt

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        if self.use_actv_ckpt:
            x = checkpoint_sequential(self.trf_blocks, self.cfg["n_layers"], x, use_reentrant=False)
        else:
            x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
