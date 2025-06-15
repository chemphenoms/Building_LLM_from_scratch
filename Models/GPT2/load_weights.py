from transformers import GPT2Model
import numpy as np
import torch

# === Reliable mapping from desired GPT-2 size to HuggingFace model identifier ===
hf_mapping = {
    "124M": "openai-community/gpt2",
    "355M": "openai-community/gpt2-medium",
    "774M": "openai-community/gpt2-large",
    "1.5B": "openai-community/gpt2-xl"
}

def assign_check(left: torch.nn.Parameter, right: torch.Tensor) -> torch.nn.Parameter:
    """
    Safely copy HF weights into our custom model.
    - Ensures shapes match exactly.
    - Detaches and clones data to avoid unexpected shared memory.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch! Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach().to(left.device))

def load_weights(gpt, gpt_hf: GPT2Model, config: dict):
    """
    Transfer all weights from HF GPT-2 to our custom model.
    Maintains exact behavior (all layers and parameters).
    """
    d = gpt_hf.state_dict()

    # === Token & position embeddings ===
    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])

    # === Transformer blocks ===
    for b in range(config["n_layers"]):
        # HF packs Q/K/V in one matrix; we split into three
        qkv_w = d[f"h.{b}.attn.c_attn.weight"]  # shape: [in, 3*out]
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=-1)
        # Transpose because HF uses [out, in], our model uses [in, out]
        gpt.trf_blocks[b].att.W_query.weight = assign_check(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign_check(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign_check(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        # Similarly for Q/K/V biases
        qkv_b = d[f"h.{b}.attn.c_attn.bias"]
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign_check(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign_check(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign_check(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        # Attention output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign_check(
            gpt.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign_check(
            gpt.trf_blocks[b].att.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"]
        )

        # === Feed-forward layers ===
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(
            gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(
            gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(
            gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(
            gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"]
        )

        # === Layer normalization parameters ===
        gpt.trf_blocks[b].norm1.weight = assign_check(
            gpt.trf_blocks[b].norm1.weight, d[f"h.{b}.ln_1.weight"]
        )
        gpt.trf_blocks[b].norm1.bias = assign_check(
            gpt.trf_blocks[b].norm1.bias, d[f"h.{b}.ln_1.bias"]
        )
        gpt.trf_blocks[b].norm2.weight = assign_check(
            gpt.trf_blocks[b].norm2.weight, d[f"h.{b}.ln_2.weight"]
        )
        gpt.trf_blocks[b].norm2.bias = assign_check(
            gpt.trf_blocks[b].norm2.bias, d[f"h.{b}.ln_2.bias"]
        )

    # === Final output norm and projection ===
    gpt.final_norm.weight = assign_check(
        gpt.final_norm.weight, d["ln_f.weight"]
    )
    gpt.final_norm.bias = assign_check(
        gpt.final_norm.bias, d["ln_f.bias"]
    )
    gpt.out_head.weight = assign_check(
        gpt.out_head.weight, d["wte.weight"]
    )

def load_hf_weights(model, num_params: str, config: dict):
    """
    Main entrypoint:
    - Loads HF GPT-2 of desired size
    - Transfers weights into our custom model
    """
    if num_params not in hf_mapping:
        raise ValueError(f"No GPT-2 model exists for size '{num_params}'. Options: {list(hf_mapping.keys())}")

    # Load pretrained HuggingFace GPT-2 in eval mode
    gpt_hf = GPT2Model.from_pretrained(hf_mapping[num_params], cache_dir="hf_checkpoints")
    gpt_hf.eval()

    # Copy weights
    load_weights(model, gpt_hf, config)