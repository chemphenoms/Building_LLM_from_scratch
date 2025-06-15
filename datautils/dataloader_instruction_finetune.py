import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import tiktoken
import utils
from datautils.dataset_instruction_finetune import InstructionDataset  # Assumes your dataset class is here


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None
):
    """
    Collate function for instruction-tuning tasks that pads sequences and prepares input-target pairs.
    """

    # Determine the maximum length in the batch (plus one for <|endoftext|>)
    batch_max_length = max(len(item) + 1 for instruction_length, item in batch)

    inputs_list, targets_list = [], []

    for instruction_length, item in batch:
        # Append <|endoftext|>
        item = item + [pad_token_id]
        padded = item + [pad_token_id] * (batch_max_length - len(item))

        inputs = torch.tensor(padded[:-1])   # All except the last token
        targets = torch.tensor(padded[1:])   # All except the first token

        # Replace all padding tokens (except the first one) in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Ignore instruction portion of the sequence
        targets[:instruction_length - 1] = ignore_index

        # Optional truncation
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    return torch.stack(inputs_list), torch.stack(targets_list)


class DataloaderIF:
    """
    Handles dataloader creation for instruction fine-tuning (e.g., Alpaca dataset).
    """

    def __init__(self,
                 tokenizer,
                 batch_size,
                 max_length,
                 dataset_name="alpaca",
                 run_type="single_gpu",
                 train_ratio=0.90,
                 collate_func=None):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.run_type = run_type
        self.collate_func = collate_func or custom_collate_fn
        self.dataset_name = dataset_name.lower()

        if self.dataset_name not in ["alpaca"]:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")

    def create_dataloader(self, data, shuffle=True, drop_last=True, num_workers=0):
        """
        Returns a DataLoader instance with or without distributed sampling.
        """
        if self.dataset_name == "alpaca":
            dataset = InstructionDataset(data, self.tokenizer)
        else:
            raise NotImplementedError("Only 'alpaca' dataset is currently supported.")

        if self.run_type == "multi_gpu":
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                drop_last=drop_last,
                sampler=DistributedSampler(dataset),
                collate_fn=self.collate_func
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=self.collate_func
            )

    def create_dataloaders(self, data, num_workers=0):
        """
        Splits the input data into training and validation sets and returns their loaders.
        """
        if not isinstance(data, list):
            raise TypeError("Data must be a list of instruction-format samples.")

        split_idx = int(self.train_ratio * len(data))
        train_data, val_data = data[:split_idx], data[split_idx:]

        train_loader = self.create_dataloader(train_data, shuffle=True, drop_last=True, num_workers=num_workers)
        val_loader = self.create_dataloader(val_data, shuffle=False, drop_last=False, num_workers=num_workers)

        return train_loader, val_loader

    def get_total_steps_epoch(self, data_files):
        """
        Calculates the total number of training steps per epoch.
        """
        total_steps = 0

        for file_path in data_files:
            data = utils.read_json_file(file_path)
            train_loader, _ = self.create_dataloaders(data, num_workers=2)
            total_steps += len(train_loader)

        return total_steps


# Example usage (optional block)
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    
    json_path = "./data/instruction-data-alpaca.json"
    data = utils.read_json_file(json_path)

    loader = DataloaderIF(
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512,
        dataset_name="alpaca",
        run_type="single_gpu",
        collate_func=custom_collate_fn
    )

    train_loader, val_loader = loader.create_dataloaders(data)

    print(f"Train Batches: {len(train_loader)}, Validation Batches: {len(val_loader)}")
