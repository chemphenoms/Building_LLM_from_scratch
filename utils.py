# utils.py
# ----------------------------------------
# Utility functions shared across modules
# ----------------------------------------

# Standard library imports
import random
import json
import numpy as np
from enum import Enum
from pathlib import Path

# Torch and numerical libraries
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# HuggingFace authentication
from huggingface_hub import login

# Custom logger
from logger import setup_logger
logger = setup_logger('utils')

# ----------------------------------------
# Global Constants and Mappings
# ----------------------------------------

# Maps data types to their byte sizes for memory estimation
datasize_mapping = {
    "fp32": 4,   # 32-bit float → 4 bytes
    "fp16": 2,   # 16-bit float → 2 bytes
    "bf16": 2    # bfloat16 → 2 bytes
}

# Maps string types to PyTorch dtypes
datatype_mapping = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

# Supported model types and their sizes
model_params_mapping = {
    "GPT2":    ["124M", "355M", "774M", "1.5B"],
    "llama2":  ["7B"],
    "llama3":  ["8B"],
    "llama3_1": ["8B"],
    "llama3_2": ["1B"]
}

# ----------------------------------------
# Reproducibility
# ----------------------------------------
def set_seed(seed: int = 123):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

    # Ensures deterministic behavior (useful for debugging)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ----------------------------------------
# Tokenization Utilities
# ----------------------------------------
def text_to_token_ids(text: str, tokenizer, cfg: dict) -> torch.Tensor:
    """
    Converts a text string into token IDs using a tokenizer.
    Adds a batch dimension.
    """
    encoded = tokenizer.encode(text, allowed_special={cfg["eos_text"]})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """
    Converts token ID tensor back into human-readable text.
    Removes the batch dimension.
    """
    return tokenizer.decode(token_ids.squeeze(0).tolist())

# ----------------------------------------
# File Readers
# ----------------------------------------
def read_text_file(file_path: Path) -> str:
    """
    Reads plain text from a file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_json_file(file_path: Path) -> dict:
    """
    Reads JSON-formatted file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# ----------------------------------------
# Model Information
# ----------------------------------------
def get_num_params(model: torch.nn.Module) -> int:
    """
    Returns the total number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters())

def get_total_size(num_params: int, data_type: str) -> float:
    """
    Estimates total model size in GB based on number of parameters and data type.
    Includes parameter memory and optimizer states (assuming Adam: 4N memory).
    """
    assert data_type in datasize_mapping, f"Unsupported data type: {data_type}"
    datatype_size = datasize_mapping[data_type]

    logger.info(f"Each parameter uses {datatype_size} bytes for type '{data_type}'.")

    logger.info("Using Adam optimizer: memory = 4N (1N for params, 1N for gradients, 2N for optimizer states)")

    total_bytes = 4 * num_params * datatype_size
    total_gb = total_bytes / (1024 ** 3)

    logger.info(f"Estimated model memory size: {total_gb:.2f} GB (excluding activations).")
    logger.info("Activations used in backpropagation are not included. Consider using activation checkpointing.")
    return total_gb

def model_memory_size(model: torch.nn.Module, input_dtype: torch.dtype = torch.float32) -> float:
    """
    Dynamically calculates the memory size of a model including parameters, gradients, and buffers.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())

    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_bytes = (total_params + total_grads + total_buffers) * element_size
    total_gb = total_bytes / (1024 ** 3)

    logger.info(f"Estimated runtime model memory: {total_gb:.2f} GB (includes grads and buffers).")
    return total_gb

# ----------------------------------------
# GPU Memory Debug Utilities
# ----------------------------------------
def start_memory_tracking():
    """
    Resets CUDA memory stats to start fresh tracking.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.warning("CUDA is not available. Skipping GPU memory tracking.")

def print_memory_usage():
    """
    Prints peak memory used on GPU (in GB).
    """
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"🔍 Max GPU memory used: {max_mem:.2f} GB")
    else:
        logger.warning("CUDA not available.")

# ----------------------------------------
# Plotting Utility
# ----------------------------------------
def plot_losses(epochs_seen: list, tokens_seen: list, train_losses: list, val_losses: list, output_dir: Path):
    """
    Plots training and validation loss vs epochs, with secondary x-axis for tokens seen.
    """
    fig, ax1 = plt.subplots()

    ax1.plot(epochs_seen, train_losses, label="Training Loss", color='blue')
    ax1.plot(epochs_seen, val_losses, linestyle="--", label="Validation Loss", color='orange')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Tokens seen on second x-axis
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Just to align the axis
    ax2.set_xlabel("Tokens Seen")

    fig.tight_layout()
    plt.savefig(output_dir / "losses.pdf")
    plt.close(fig)  # Avoid memory leak if many plots are generated

# ----------------------------------------
# HuggingFace Login
# ----------------------------------------
def login_hf():
    """
    Logs into Hugging Face using access token stored in `config_hf.json`.
    """
    try:
        with open("config_hf.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        access_token = config.get("HF_ACCESS_TOKEN", None)
        assert access_token, "HF_ACCESS_TOKEN not found in config."

        login(token=access_token)
        logger.info("Logged into Hugging Face Hub.")

    except FileNotFoundError:
        logger.error("'config_hf.json' not found. Please create the file with your access token.")
    except Exception as e:
        logger.error(f"Error logging into Hugging Face: {e}")
