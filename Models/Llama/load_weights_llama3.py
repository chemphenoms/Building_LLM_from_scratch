import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def assign(left, right, tensor_name="unknown"):
    """
    Safely assigns a weight tensor 'right' to a model parameter 'left'.
    Checks for shape compatibility and wraps in torch.nn.Parameter.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    """
    Loads pre-trained weights from a parameter dictionary into the custom LLaMA model.
    This version uses Meta's LLaMA 3.x safetensors parameter naming.
    """
    # Load token embedding weights
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        # --- Attention Weights ---
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # --- Feedforward Network Weights ---
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # --- Final Norm Layer ---
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    # --- Output Head (with or without weight tying) ---
    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")


def load_hf_weights(model, model_name, config):
    """
    Downloads and loads weights for LLaMA 3.x models from HuggingFace Hub using safetensors.
    Supported model variants:
        - "llama3":        Meta-LLaMA-3-8B
        - "llama3_1":      LLaMA-3.1-8B
        - "llama3_2":      LLaMA-3.2-1B
    """
    combined_weights = {}

    if model_name == "llama3":
        for i in range(1, 5):
            weights_file = hf_hub_download(
                repo_id="meta-llama/Meta-Llama-3-8B",
                filename=f"model-0000{i}-of-00004.safetensors",
                local_dir="Llama-3-8B"
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    elif model_name == "llama3_1":
        for i in range(1, 5):
            weights_file = hf_hub_download(
                repo_id="meta-llama/Llama-3.1-8B",
                filename=f"model-0000{i}-of-00004.safetensors",
                local_dir="Llama-3.1-8B"
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    elif model_name == "llama3_2":
        weights_file = hf_hub_download(
            repo_id="meta-llama/Llama-3.2-1B",
            filename="model.safetensors",
            local_dir="Llama-3.2-1B"
        )
        combined_weights = load_file(weights_file)

    # Load weights into the model and clean up
    load_weights_into_llama(model, config, combined_weights)
    del combined_weights  # Free memory
