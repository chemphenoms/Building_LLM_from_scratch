import torch
from huggingface_hub import hf_hub_download

def assign(left, right):
    """
    Safely assigns weights from 'right' (loaded) to 'left' (model parameter).
    Ensures shape compatibility and wraps the tensor as a torch.nn.Parameter.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    """
    Loads pretrained weights from 'params' into the custom 'model' according to 'param_config'.
    This function assumes the parameter naming follows HuggingFace LLaMA naming conventions.
    """
    
    # Load token embedding weights
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):
        # ---- Attention Block Weights ----
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"]
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # ---- Feedforward Block Weights ----
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # ⚠️ Note: w2 and w3 appear swapped in the HF weight file, so we assign carefully.
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # ---- Final Output & Normalization ----
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


def load_hf_weights(model, config):
    """
    Downloads and loads weights from HuggingFace Hub (Meta's LLaMA-2-7B) into the model.
    Uses HuggingFace's `hf_hub_download` utility and assumes a local directory for caching.
    """
    # Downloads the .pth weight file from HuggingFace (cache if already downloaded)
    weights_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="consolidated.00.pth",
        local_dir="Llama-2-7b"  # Can be customized
    )

    # Load the weights using PyTorch, with memory optimization
    weights = torch.load(weights_file, weights_only=True)

    # Now assign all weights to the model
    load_weights_into_llama(model, config, weights)
