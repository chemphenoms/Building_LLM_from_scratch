# === GPT-2 Configuration Setup ===
# We define configurations (settings) for different GPT-2 model sizes, like small (124M) to large (1.5B).
# These settings include how much vocabulary it knows, how deep the network is, and how many "brains" (attention heads) it uses.

# Common values used across all model sizes to avoid repetition.
DEFAULT_VOCAB_SIZE = 50257         # Total number of words/subwords the model can understand.
DEFAULT_CONTEXT_LENGTH = 1024      # How many words/tokens the model can consider at once (like its short-term memory size).
DEFAULT_DROPOUT = 0.1              # Dropout rate to prevent overfitting (randomly turns off neurons during training).
DEFAULT_QKV_BIAS = False           # Whether to add bias in the attention mechanism. False means no additional bias is used.
EOS_ID = 50256                     # Special ID indicating the "end of a sentence" or "end of text".
EOS_TEXT = "<|endoftext|>"         # Human-readable form of that end-of-text marker.

# This function helps us build a configuration dictionary for a GPT-2 model,
# given its specific size-related parameters: embedding dimension, number of heads, and layers.
def build_gpt_config(emb_dim, n_heads, n_layers):
    return {
        "vocab_size": DEFAULT_VOCAB_SIZE,          # Shared setting across all sizes
        "context_length": DEFAULT_CONTEXT_LENGTH,  # How far back the model can look in the input
        "emb_dim": emb_dim,                        # Size of internal word representation (higher = more expressive)
        "n_heads": n_heads,                        # Like parallel mini-brains; more heads = better multitasking
        "n_layers": n_layers,                      # Depth of the model; more layers = more complexity
        "drop_rate": DEFAULT_DROPOUT,              # Dropout to improve generalization
        "qkv_bias": DEFAULT_QKV_BIAS,              # Bias setting in attention layers
        "eos_id": EOS_ID,                          # Special token ID marking end of sequence
        "eos_text": EOS_TEXT                       # Textual marker for end-of-sequence
    }

# Dictionary of all available GPT-2 model sizes with their respective configurations.
# The keys are names based on number of parameters (in millions or billions).
available_configs = {
    "124M": build_gpt_config(768, 12, 12),         # Small GPT-2 (124 million parameters)
    "355M": build_gpt_config(1024, 16, 24),        # Medium GPT-2 (355 million parameters)
    "774M": build_gpt_config(1280, 20, 36),        # Large GPT-2 (774 million parameters)
    "1.5B": build_gpt_config(1600, 25, 48),        # Extra-large GPT-2 (1.5 billion parameters)
}

# Function to get the configuration of a GPT-2 model based on the model size requested by the user.
def get_config_gpt2(num_params):
    num_params = str(num_params)  # Ensure input is string, as keys in dictionary are string-based

    # Check if the requested model size exists in the available configurations
    if num_params not in available_configs:
        # If not, raise a clear error message with options
        raise ValueError(
            f"GPT-2 config for model '{num_params}' not found. "
            f"Available options: {list(available_configs.keys())}"
        )
    
    # Return the configuration dictionary corresponding to the selected model
    return available_configs[num_params]
