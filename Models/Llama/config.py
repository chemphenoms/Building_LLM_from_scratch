import torch
from Models.Llama.common_components import rescale_theta

# -----------------------------------------------------------
# Model Configuration Dictionaries for LLaMA Models
# -----------------------------------------------------------

LLAMA2_CONFIG_7B = {
    "vocab_size": 32_000,           # Standard vocabulary size for LLaMA 2
    "context_length": 4096,         # Sequence length supported
    "emb_dim": 4096,                # Embedding size
    "n_heads": 32,                  # Number of self-attention heads
    "n_layers": 32,                 # Number of transformer layers
    "hidden_dim": 11_008,           # Intermediate hidden layer in FFN
    "dtype": torch.bfloat16         # Precision for model weights and activations
}

LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,          # Expanded vocabulary for multilingual/general use
    "context_length": 8192,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14_336,
    "n_kv_groups": 8,               # Grouped-query attention support
    "rope_base": 500_000.0,         # RoPE base frequency parameter
    "rope_freq": None,              # Optional override for RoPE frequency scaling
    "dtype": torch.bfloat16,
    "eos_id": 128_001,
    "eos_text": "<|end_of_text|>"
}

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,
    "context_length": 131_072,      # Extended context length
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14_336,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "eos_id": 128_001,
    "eos_text": "<|end_of_text|>"
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,                # Halved embedding dimension for lighter variant
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "eos_id": 128_001,
    "eos_text": "<|end_of_text|>"
}

# -----------------------------------------------------------
# Available Configs for Retrieval by Parameter Size
# -----------------------------------------------------------

available_configs_llama2 = {
    "7B": LLAMA2_CONFIG_7B
}

available_configs_llama3 = {
    "8B": LLAMA3_CONFIG_8B
}

available_configs_llama3_1 = {
    "8B": LLAMA31_CONFIG_8B
}

available_configs_llama3_2 = {
    "1B": LLAMA32_CONFIG_1B
}

# -----------------------------------------------------------
# Configuration Loader Functions
# -----------------------------------------------------------

def get_config_llama(num_params, model_name):
    """
    Retrieves the appropriate LLaMA configuration and optionally rescales RoPE base.

    Args:
        num_params (str or int): Number of model parameters (e.g., '7B').
        model_name (str): Model version identifier (e.g., 'llama2', 'llama3').

    Returns:
        dict: Model configuration dictionary.
    """
    num_params = str(num_params)
    available_configs = globals().get(f"available_configs_{model_name}", None)

    assert available_configs is not None, f"No configuration found for model '{model_name}'"
    assert num_params in available_configs, f"A {model_name} model with {num_params} parameters does not exist."

    config = available_configs[num_params]
    old_context_length = config["context_length"]

    if old_context_length != 1024:
        config["context_length"] = 1024
        config["rope_base"] = rescale_theta(
            config["rope_base"],
            old_context_length,
            config["context_length"]
        )
        print("New RoPE theta:", config["rope_base"])

    return config

# ✅ Enhancement:
# The above hardcoded downscaling to 1024 could be made parameterized via an argument:
# def get_config_llama(num_params, model_name, target_context_length=1024):


def get_config_llama2(num_params):
    """
    Retrieves the LLaMA2 configuration based on the number of parameters.
    """
    num_params = str(num_params)
    assert num_params in available_configs_llama2, "A LLaMA2 model with given number of parameters does not exist."
    return available_configs_llama2[num_params]


def get_config_llama3():
    """
    Directly returns the base LLaMA3-8B configuration.
    """
    return LLAMA3_CONFIG_8B


def get_config_llam31():
    """
    Returns adjusted LLaMA3.1-8B config with a reduced context length and rescaled RoPE base.
    """
    old_context_length = LLAMA31_CONFIG_8B["context_length"]
    LLAMA31_CONFIG_8B["context_length"] = 1024

    LLAMA31_CONFIG_8B["rope_base"] = rescale_theta(
        LLAMA31_CONFIG_8B["rope_base"],
        old_context_length,
        LLAMA31_CONFIG_8B["context_length"]
    )
    print("New RoPE theta:", LLAMA31_CONFIG_8B["rope_base"])
    return LLAMA31_CONFIG_8B


def get_config_llam32():
    """
    Returns adjusted LLaMA3.2-1B config with reduced context length and rescaled RoPE base.
    """
    old_context_length = LLAMA32_CONFIG_1B["context_length"]
    LLAMA32_CONFIG_1B["context_length"] = 1024

    LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
        LLAMA32_CONFIG_1B["rope_base"],
        old_context_length,
        LLAMA32_CONFIG_1B["context_length"]
    )
    print("New RoPE theta:", LLAMA32_CONFIG_1B["rope_base"])
    return LLAMA32_CONFIG_1B
