# ==============================
#         IMPORT BLOCK
# ==============================

import tiktoken
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch.distributed import (
    init_process_group,
    destroy_process_group,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig, 
    StateDictType,
    ShardingStrategy,
    BackwardPrefetch
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
    enable_wrap,
    wrap,
)

# Internal utility and model imports
import utils
from Models.GPT2.config import get_config_gpt2
from Models.Llama.config import get_config_llama
from Models.GPT2.GPT2 import GPTModel, TransformerBlock
from Models.Llama.Llama2 import Llama2Model, Llama2Tokenizer
from Models.Llama.Llama3 import Llama3Model, Llama3Tokenizer
from Models.Llama.common_components import rescale_theta
from huggingface_hub import hf_hub_download, login
from logger import setup_logger

logger = setup_logger('build_components')


# ==============================
#      CONFIGURATION BUILDER
# ==============================

def build_config(args):
    """
    Constructs and returns model configuration dictionary.

    Args:
        args: Command-line arguments.
    
    Returns:
        config: Configuration dictionary with model hyperparameters.
    """
    config = None

    if args.model == "GPT2":
        config = get_config_gpt2(args.num_params)
    elif args.model.startswith("llama"):
        config = get_config_llama(args.num_params, args.model)
    
    config.update({"dtype": utils.datatype_mapping[args.data_type]})

    if args.load_weights and args.model == "GPT2":
        config.update({"qkv_bias": True})

    if args.debug:
        config.update({
            "context_length": 10,
            "emb_dim": 32,
            "hidden_dim": 10,
            "n_heads": 16,
            "n_layers": 2,
            "qkv_bias": False
        })

    return config


# ==============================
#         LOAD WEIGHTS
# ==============================

def load_model_weights(args, config, model):
    """
    Load pretrained weights from HuggingFace based on model type.

    Args:
        args: Command-line arguments.
        config: Model config.
        model: Model instance.
    """
    utils.login_hf()

    if args.model == "GPT2":
        from Models.GPT2.load_weights import load_hf_weights
        load_hf_weights(model, args.num_params, config)

    elif args.model == "llama2":
        from Models.Llama.load_weights_llama2 import load_hf_weights
        load_hf_weights(model, config)

    elif args.model.startswith("llama3"):
        from Models.Llama.load_weights_llama3 import load_hf_weights
        load_hf_weights(model, args.model, config)


# ==============================
#         LORA INTEGRATION
# ==============================

def activate_lora(args, cfg, model):
    """
    Modify model for LoRA training.

    Args:
        args: Command-line arguments.
        cfg: Config dictionary.
        model: Model instance.
    """
    from lora import replace_linear_with_lora

    logger.info("Using LoRA...")
    for param in model.parameters():
        param.requires_grad = False

    replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dtype=cfg["dtype"])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable LoRA parameters: {trainable_params:,}")


# ==============================
#         DISTRIBUTED SETUP
# ==============================

def multigpu_setup(args, rank, model):
    """
    Setup model for multi-GPU training using FSDP or DDP.

    Args:
        args: Command-line arguments.
        rank: Rank of process.
        model: Model instance.

    Returns:
        model: Wrapped model.
    """
    if torch.cuda.is_available():
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "cpu_offload": None,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "mixed_precision": None,
            "sync_module_states": False,
            "device_id": torch.cuda.current_device(),
            "use_orig_params": False
        }

        if args.use_fsdp:
            from datautils.mixed_precision import fpSixteen, bfSixteen
            if args.mixed_precision:
                fsdp_kwargs["mixed_precision"] = fpSixteen if args.mixed_precision == "fp16" else bfSixteen
            if args.use_lora:
                fsdp_kwargs["use_orig_params"] = True

            auto_wrap_policy = ModuleWrapPolicy(module_classes=[nn.Embedding, TransformerBlock])

            model = FSDP(model, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
        else:
            model = DDP(model, device_ids=[rank])

        if rank == 0:
            logger.info("Model wrapped with DDP/FSDP.")
            utils.print_memory_usage()

    return model


# ==============================
#         MODEL CREATION
# ==============================

def build_model(config, rank, device, args):
    """
    Construct and prepare the model with optional FSDP, LoRA, weight loading.

    Returns:
        model: Finalized model object.
    """
    utils.start_memory_tracking()

    if args.model == "GPT2":
        model = GPTModel(config, args.use_actv_ckpt)
    elif args.model == "llama2":
        model = Llama2Model(config, args.use_actv_ckpt)
    elif args.model.startswith("llama3"):
        model = Llama3Model(config, args.use_actv_ckpt)
    else:
        raise Exception("Unsupported model specified.")

    if rank == 0:
        logger.info(f"Total parameters: {utils.get_num_params(model):,}")
        utils.model_memory_size(model, config["dtype"])

    if args.load_weights:
        if rank != 0 and args.run_type == "multi_gpu":
            torch.distributed.barrier()
        load_model_weights(args, config, model)
        if rank == 0 and args.run_type == "multi_gpu":
            torch.distributed.barrier()

    if args.use_lora:
        activate_lora(args, config, model)

    if not args.use_fsdp:
        model.to(device)

    if rank == 0:
        logger.info("Model moved to CUDA.")
        utils.print_memory_usage()

    if args.use_zero_opt:
        for p in model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

    if args.run_type == "multi_gpu":
        model = multigpu_setup(args, rank, model)

    return model


# ==============================
#         OPTIMIZER SETUP
# ==============================

def build_optimizer(args, model):
    """
    Create optimizer (AdamW or ZeroRedundancy).

    Returns:
        optimizer: Optimizer object.
    """
    if args.use_zero_opt:
        return ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=0.1
        )
    else:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)


# ==============================
#         TOKENIZER SETUP
# ==============================

def build_tokenizer(rank, args):
    """
    Load tokenizer based on the model type.

    Returns:
        tokenizer: Tokenizer object.
    """
    if rank != 0 and args.run_type == "multi_gpu":
        torch.distributed.barrier()

    utils.login_hf()

    if args.model == "GPT2":
        tokenizer = tiktoken.get_encoding("gpt2")

    elif args.model == "llama2":
        tokenizer_file = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model", local_dir="Llama-2-7b")
        tokenizer = Llama2Tokenizer(tokenizer_file)

    elif args.model.startswith("llama3"):
        model_mapping = {
            "llama3": ("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
            "llama3_1": ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
            "llama3_2": ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
        }
        repo_id, local_dir = model_mapping[args.model]
        tokenizer_file = hf_hub_download(repo_id=repo_id, filename="original/tokenizer.model", local_dir=local_dir)
        tokenizer = Llama3Tokenizer(tokenizer_file)

    else:
        raise Exception("Tokenizer not found for this model type.")

    if rank == 0 and args.run_type == "multi_gpu":
        torch.distributed.barrier()

    return tokenizer


# ==============================
#      COMPONENTS WRAPPER
# ==============================

def build_components(rank: int, device: torch.device, args):
    """
    Build training components.

    Returns:
        Tuple: (config, model, optimizer, tokenizer)
    """
    try:
        config = build_config(args)
        model = build_model(config, rank, device, args)
        optimizer = build_optimizer(args, model)
        tokenizer = build_tokenizer(rank, args)

        return config, model, optimizer, tokenizer

    except Exception as e:
        logger.exception(f"Exception while building components: {e}")
