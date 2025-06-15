import argparse
import warnings
import os
import torch
from utils import model_params_mapping


def perform_checks(args):
    """
    Performs critical configuration validations to ensure compatibility and prevent runtime errors.
    """

    if not args.warnings:
        warnings.filterwarnings("ignore")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")

    if args.num_params not in model_params_mapping.get(args.model, []):
        raise ValueError(
            f"Unsupported model configuration: {args.model} with {args.num_params}. "
            f"Supported sizes: {model_params_mapping.get(args.model, [])}"
        )

    if args.run_type == "single_gpu" and args.use_fsdp:
        raise ValueError("FSDP requires multi-GPU. It's not supported for single GPU training.")

    if args.use_zero_opt and args.use_fsdp:
        raise ValueError("Zero Redundancy Optimizer cannot be used with FSDP.")

    if args.use_fsdp and not torch.cuda.is_available():
        raise EnvironmentError("FSDP requires CUDA devices. Please ensure CUDA is available.")

    if not args.use_fsdp and args.mixed_precision:
        raise ValueError("Mixed precision can only be enabled with FSDP.")


def get_args():
    """
    Parses and validates command-line arguments for training configuration.
    """

    parser = argparse.ArgumentParser(description="Large Language Model Training Configuration")

    # Dataset and I/O paths
    parser.add_argument('--data_dir', type=str, default='/home/ec2-user/train-llm-from-scratch/Datasets/Gutenberg/data_dir_small',
                        help='Path to the dataset directory.')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory to save model checkpoints.')

    # Training configuration
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Base learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps.')
    parser.add_argument('--initial_lr', type=float, default=1e-5, help='Initial learning rate before warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')

    # Logging & Evaluation
    parser.add_argument('--print_sample_iter', type=int, default=10, help='Steps between printing sample outputs.')
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency (in steps).')
    parser.add_argument('--save_ckpt_freq', type=int, default=100, help='Checkpoint save frequency (in steps).')

    # Model Configuration
    parser.add_argument('--model', type=str, default="GPT2", choices=["GPT2", "llama2", "llama3", "llama3_1", "llama3_2"],
                        help='Target model architecture.')
    parser.add_argument('--num_params', type=str, default="124M", help='Model size identifier.')
    parser.add_argument('--load_weights', action="store_true", help='Load pretrained weights if available.')
    parser.add_argument('--debug', action="store_true", help='Use a small model for debugging purposes.')

    # Hardware/Precision/Optimization
    parser.add_argument('--run_type', type=str, default="single_gpu", choices=['single_gpu', 'multi_gpu'],
                        help="Run type: single or multi-GPU.")
    parser.add_argument('--use_fsdp', action="store_true", help='Enable Fully Sharded Data Parallelism.')
    parser.add_argument('--use_zero_opt', action="store_true", help='Enable Zero Redundancy Optimizer (not with FSDP).')
    parser.add_argument('--use_actv_ckpt', action="store_true", help='Enable activation checkpointing.')
    parser.add_argument('--data_type', type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Model precision data type.")
    parser.add_argument('--mixed_precision', type=str, choices=["fp16", "bf16"],
                        help="Enable mixed precision (only works with FSDP).")

    # Fine-tuning & Dataset
    parser.add_argument('--finetune', action="store_true", help='Enable fine-tuning mode.')
    parser.add_argument('--dataset', type=str, default="gutenberg", choices=["gutenberg", "alpaca"],
                        help='Dataset name.')

    # LoRA (Low-Rank Adaptation)
    parser.add_argument('--use_lora', action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha.')

    # Warnings & Logs
    parser.add_argument('--warnings', action="store_true", help='Enable Python warnings.')

    args = parser.parse_args()

    perform_checks(args)

    return args


# Optional CLI entry point
if __name__ == "__main__":
    args = get_args()
    print("Arguments parsed and validated successfully:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
