import torch
from torch.utils.data import Dataset
import tiktoken


def format_input(entry):
    """
    Format instruction + input for Alpaca-style datasets.

    Example output:
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    <instruction text>

    ### Input:
    <input text>
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def format_input_phi(entry):
    """
    Format instruction + input for Phi-style datasets.

    Example output:
    <|user|>
    <instruction>
    <input>

    <|assistant|>:
    <response>
    """
    instruction_text = f"<|user|>\n{entry['instruction']}"
    input_text = f"\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """
    Dataset for instruction-tuning using Alpaca-style JSON format.
    
    Each item returns:
        - Length of prompt part (instruction + input)
        - Full tokenized sequence (including response)
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        self.instruction_lengths = []

        for entry in data:
            # Format and tokenize
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            encoded = tokenizer.encode(full_text)
            self.encoded_texts.append(encoded)

            # Also store where the instruction part ends
            instr_len = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instr_len)

    def __getitem__(self, index):
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    """
    Dataset for instruction-tuning using Phi-style conversational format.
    
    Each item returns:
        - Tokenized instruction + input + response
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            encoded = tokenizer.encode(full_text)
            self.encoded_texts.append(encoded)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# Optional usage example (for unit testing)
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    
    sample_data = [
        {
            "instruction": "Translate the following sentence to French.",
            "input": "I love machine learning.",
            "output": "J'aime l'apprentissage automatique."
        },
        {
            "instruction": "Summarize the paragraph.",
            "input": "",
            "output": "The paragraph discusses the importance of AI in healthcare."
        }
    ]

    alpaca_ds = InstructionDataset(sample_data, tokenizer)
    phi_ds = InstructionDatasetPhi(sample_data, tokenizer)

    print("Sample from Alpaca-style dataset:")
    print(alpaca_ds[0])
    print("Sample from Phi-style dataset:")
    print(phi_ds[0])
