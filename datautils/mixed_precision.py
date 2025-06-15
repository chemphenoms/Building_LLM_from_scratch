import torch
from torch.distributed.fsdp import MixedPrecision


# ---------------------------- #
# FSDP Mixed Precision Policies
# ---------------------------- #

# Requires GradScaler during training loop (fp16 needs loss scaling)
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,       # Parameters stored in FP16
    reduce_dtype=torch.float16,      # Gradients reduced in FP16
    buffer_dtype=torch.float16       # Buffers (like batch norm stats) in FP16
)

# Preferred for newer hardware (e.g. A100, H100, 3090, etc.)
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # Parameters stored in BF16
    reduce_dtype=torch.bfloat16,     # Gradients reduced in BF16
    buffer_dtype=torch.bfloat16      # Buffers in BF16
)

# Hybrid: Model weights in FP32, communication in BF16
bf16_hybrid_policy = MixedPrecision(
    param_dtype=torch.float32,       # Parameters stored in FP32 for higher accuracy
    reduce_dtype=torch.bfloat16,     # Gradients reduced in BF16
    buffer_dtype=torch.bfloat16      # Buffers in BF16
)

# Full Precision: Baseline (useful for debugging or precision-critical tasks)
fp32_policy = MixedPrecision(
    param_dtype=torch.float32,       # Parameters in FP32
    reduce_dtype=torch.float32,      # Gradients in FP32
    buffer_dtype=torch.float32       # Buffers in FP32
)

# ---------------------------- #
# Optional: Helper Dictionary
# ---------------------------- #

mixed_precision_policies = {
    "fp16": fp16_policy,
    "bf16": bf16_policy,
    "bf16_hybrid": bf16_hybrid_policy,
    "fp32": fp32_policy
}

# ---------------------------- #
# Usage Example
# ---------------------------- #
if __name__ == "__main__":
    # Select desired policy dynamically
    precision = "bf16"  # Choose from: fp16, bf16, bf16_hybrid, fp32
    chosen_policy = mixed_precision_policies[precision]
    
    print(f"Selected FSDP policy: {precision}")
    print(chosen_policy)


# ---------------------------- #
# When using FSDP in a training loop, pass the MixedPrecision policy like so:
# ---------------------------- #

"""
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    mixed_precision=chosen_policy,
    # sharding_strategy=ShardingStrategy.FULL_SHARD, etc.
)
"""