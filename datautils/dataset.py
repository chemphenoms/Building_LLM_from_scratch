import torch
from torch.utils.data import Dataset
import tiktoken


class DatasetPT(Dataset):
    """
    PyTorch Dataset for Pretraining Causal Language Models on tokenized text.
    
    Each sample is a pair of:
      - input_ids: tokens from position i to i + max_length
      - target_ids: tokens from position i+1 to i + max_length + 1
    
    Args:
        txt (str): The entire text corpus.
        tokenizer (tiktoken.Encoding): A tokenizer like GPT-2's tiktoken encoder.
        max_length (int): Sequence length per training example.
        stride (int): Step size for sliding window over tokenized input.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the input while keeping <|endoftext|> as a special token
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # Create sliding window sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ✅ Optional usage example (can be removed if not needed)
if __name__ == "__main__":
    # Example usage with dummy text
    tokenizer = tiktoken.get_encoding("gpt2")
    dummy_text = "Once upon a time, there was a brave AI model. <|endoftext|> It solved many problems."

    dataset = DatasetPT(
        txt=dummy_text,
        tokenizer=tokenizer,
        max_length=10,
        stride=5
    )

    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        print(f"Sample {i}:")
        print(f"Input IDs: {input_ids}")
        print(f"Target IDs: {target_ids}")
