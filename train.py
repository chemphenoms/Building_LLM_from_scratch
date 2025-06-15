# trainer.py

# --------------------------------------------
# Robust Imports with Safety Checks
# --------------------------------------------
try:
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    # FSDP is available from torch >=1.12
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    from tqdm import tqdm
    from pathlib import Path
    import math

    import utils  # Internal utility functions
    from generate import generate  # Text generation logic
    from logger import setup_logger  # Logger utility

except ImportError as e:
    raise ImportError(f"Missing dependency or incorrect version: {e}. "
                      "Please install all required packages using `pip install -r requirements.txt`.")

# --------------------------------------------
# Torch version check for FSDP compatibility
# --------------------------------------------
required_version = (1, 12)
torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version < required_version:
    raise RuntimeError(f"FSDP requires PyTorch >= 1.12. Detected: {torch.__version__}")

# --------------------------------------------
# Logger Initialization
# --------------------------------------------
logger = setup_logger('train')
logger.info("Trainer initialized with all modules loaded.")

# --------------------------------------------
# Trainer Class
# --------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, config, data_files, loaderObj, save_dir,
                 warmup_steps=10, initial_lr=1e-5, min_lr=1e-6, device="cpu",
                 rank=0, eval_freq=1, save_ckpt_freq=1, print_sample_iter=1, eval_iter=1):
        """
        Initializes the Trainer object for training and evaluation.

        Parameters:
        - model: the neural network model (can be wrapped with DDP or FSDP)
        - optimizer: optimizer (e.g., AdamW)
        - config: model config dict
        - data_files: list of paths to training data
        - loaderObj: an instance of DataloaderIF or DataloaderPT
        - save_dir: path to save checkpoints
        - device: CUDA or CPU device
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_files = data_files
        self.loaderObj = loaderObj
        self.save_dir = save_dir
        self.device = device
        self.rank = rank

        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr

        self.eval_freq = eval_freq
        self.save_ckpt_freq = save_ckpt_freq
        self.print_sample_iter = print_sample_iter
        self.eval_iter = eval_iter

        self.global_step = -1
        self.tokens_seen = 0

        self.train_losses = []
        self.val_losses = []
        self.track_lrs = []
        self.track_tokens_seen = []

    # ----------------------------------------
    # Custom Training Loop
    # ----------------------------------------
    def calc_loss_batch(self, input_batch, target_batch):
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def train_batch(self, input_batch, target_batch):
        try:
            self.optimizer.zero_grad()
            self.global_step += 1

            # Learning Rate Warmup and Cosine Annealing
            if self.global_step < self.warmup_steps:
                lr = self.initial_lr + self.global_step * self.lr_increment
            else:
                progress = ((self.global_step - self.warmup_steps) / (self.total_training_steps - self.warmup_steps))
                lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.track_lrs.append(lr)

            loss = self.calc_loss_batch(input_batch, target_batch)
            loss.backward()

            # Gradient Clipping (especially useful for transformers)
            if isinstance(self.model, FSDP):
                self.model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
            elif isinstance(self.model, DDP):
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.tokens_seen += input_batch.numel()

        except Exception as e:
            logger.error(f"Error in train_batch(): {e}")

    def train_epoch(self, epoch_no, train_loader, val_loader, start_context="Every effort moves you"):
        try:
            self.model.train()
            for input_batch, target_batch in train_loader:
                self.train_batch(input_batch, target_batch)

                if self.global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(train_loader, val_loader, self.eval_iter)
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.track_tokens_seen.append(self.tokens_seen)

                    if self.rank == 0:
                        logger.info(f"Epoch {epoch_no+1} | Step {self.global_step} "
                                    f"| Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

                if self.global_step % self.print_sample_iter == 0:
                    self.generate_and_print_sample(start_context)

                if self.global_step % self.save_ckpt_freq == 0:
                    self.save_checkpoint(f"model_pg_{self.global_step}.pth")

        except Exception as e:
            logger.error(f"Error in train_epoch(): {e}")

    def train_model(self, n_epochs):
        self.peak_lr = self.optimizer.param_groups[0]["lr"]
        self.total_training_steps = self.loaderObj.get_total_steps_epoch(self.data_files) * n_epochs
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps

        if self.rank == 0:
            pbar = tqdm(total=n_epochs * len(self.data_files))

        try:
            for epoch in range(n_epochs):
                for file_path in self.data_files:
                    trailing_text = " " + self.config["eos_text"] + " "
                    raw_text = utils.read_text_file(file_path) + trailing_text

                    train_loader, val_loader = self.loaderObj.create_dataloaders(raw_text, num_workers=2)

                    if hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)

                    self.train_epoch(epoch, train_loader, val_loader)

                    if self.rank == 0:
                        pbar.update(1)

        except KeyboardInterrupt:
            self.save_checkpoint(f"model_pg_{self.global_step}_interrupted.pth")

        return self.train_losses, self.val_losses, self.track_tokens_seen, self.track_lrs

    def finetune_model(self, n_epochs):
        self.peak_lr = self.optimizer.param_groups[0]["lr"]
        self.total_training_steps = self.loaderObj.get_total_steps_epoch(self.data_files) * n_epochs
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps

        if self.rank == 0:
            pbar = tqdm(total=n_epochs * len(self.data_files))

        try:
            for epoch in range(n_epochs):
                for file_path in self.data_files:
                    json_data = utils.read_json_file(file_path)

                    train_loader, val_loader = self.loaderObj.create_dataloaders(json_data, num_workers=2)

                    if hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)

                    self.train_epoch(epoch, train_loader, val_loader,
                                     start_context="Below is an instruction that describes a task. "
                                                   "Write a response that appropriately completes the request.\n\n"
                                                   "### Instruction:\nWhat is an antonym of 'complicated'?")

                    if self.rank == 0:
                        pbar.update(1)

        except KeyboardInterrupt:
            self.save_checkpoint(f"model_pg_{self.global_step}_interrupted.pth")

        return self.train_losses, self.val_losses, self.track_tokens_seen, self.track_lrs

    def generate_and_print_sample(self, start_context, temperature=1.0, top_k=5,
                                  memory_check=True, max_new_tokens=200):
        self.model.eval()
        context_size = self.config["context_length"]

        encoded = utils.text_to_token_ids(start_context, self.loaderObj.tokenizer, self.config).to(self.device)

        token_ids = generate(
            model=self.model, idx=encoded, max_new_tokens=max_new_tokens,
            context_size=context_size, temperature=temperature, top_k=top_k, eos_id=self.config["eos_id"]
        )

        decoded = utils.token_ids_to_text(token_ids, self.loaderObj.tokenizer)
        self.model.train()

        if self.rank == 0 and memory_check:
            logger.info(f"Generated Sample: {decoded.replace(chr(10), ' ')}")

    def save_checkpoint(self, file_name):
        if isinstance(self.model, DDP):
            torch.distributed.barrier()

        try:
            file_path = self.save_dir / file_name

            if self.rank == 0:
                if isinstance(self.model, DDP):
                    torch.save(self.model.module.state_dict(), file_path)
                elif not isinstance(self.model, FSDP):
                    torch.save(self.model.state_dict(), file_path)

            if isinstance(self.model, FSDP):
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                    state = self.model.state_dict()
                if self.rank == 0:
                    torch.save(state, file_path)

            logger.info(f"Checkpoint saved: {file_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        if isinstance(self.model, DDP):
            torch.distributed.barrier()

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        num_batches = min(num_batches or len(data_loader), len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
        return total_loss / num_batches

    def evaluate_model(self, train_loader, val_loader, eval_iter=5):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, eval_iter)
            val_loss = self.calc_loss_loader(val_loader, eval_iter)
        self.model.train()
        return train_loss, val_loss
