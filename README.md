# 🚀 Large Language Model (LLM) Training Framework

This repository provides a **complete, modular implementation** for training **GPT2** and **LLaMA models (LLaMA2 and LLaMA3)** from scratch using Python and PyTorch. It supports both **pretraining on raw text** and **finetuning using instruction datasets** like Alpaca. It also includes efficient training features such as **LoRA**, **activation checkpointing**, **mixed precision**, and **FSDP (Fully Sharded Data Parallel)** to scale your models on single or multiple GPUs.

---

## 🔧 What This Repo Offers

### ✅ From Scratch Implementation

* **Models Supported**:

  * GPT2 (124M to 1.5B parameter variants)
  * LLaMA2 (7B)
  * LLaMA3 (1B and 8B)

* **Two Training Modes**:

  * **Pretraining** on unstructured text (e.g., books, articles)
  * **Finetuning** using instruction datasets (e.g., question-answering pairs)

### ✅ Training Flexibility

* **Run Types**:

  * Single GPU
  * Multi-GPU using PyTorch’s Distributed Data Parallel (DDP)

* **Memory Optimizations**:

  * **Activation Checkpointing**: Save memory during backpropagation
  * **FSDP**: Efficient multi-GPU training
  * **LoRA**: Lightweight finetuning for large models
  * **Mixed Precision Training**: Faster, less memory-intensive training

---

## 📦 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10 with CUDA
* CUDA drivers
* Hugging Face Transformers
* Packages listed in `requirements.txt`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🔐 Hugging Face Authentication

To download pretrained weights from Hugging Face:

1. Create an account on [huggingface.co](https://huggingface.co)
2. Generate an access token
3. Save it inside a config file named `config_hf.json`:

```json
{
  "HF_ACCESS_TOKEN": "your_token_here"
}
```

---

## 🖥️ Tested Hardware

* Single GPU: `g4dn.xlarge` (AWS)
* Multi-GPU: `g4dn.12xlarge` (AWS)

---

## 🚀 Getting Started

### 📚 Download Datasets

#### 1. Gutenberg Dataset (for pretraining)

```bash
cd Datasets/Gutenberg
bash setup.sh
```

#### 2. Alpaca Dataset (for instruction finetuning)

```bash
cd Datasets/Alpaca
bash setup.sh
```

---

## 🔄 Training Examples

### 🔹 Pretrain GPT2 on Raw Text

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --load_weights
```

### 🔹 Multi-GPU Training with LoRA (Low-Rank Adaptation)

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --model GPT2 --run_type multi_gpu --use_lora --load_weights
```

### 🔹 Add Activation Checkpointing

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --model GPT2 --num_params 774M --run_type multi_gpu --use_lora --use_actv_ckpt --load_weights
```

### 🔹 Use Fully-Sharded Data Parallelism (FSDP)

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --model GPT2 --num_params 774M --run_type multi_gpu --use_lora --use_actv_ckpt --use_fsdp --load_weights
```

### 🔹 Finetune LLaMA3.2 (1B) on Alpaca Dataset

```bash
python main.py --dataset alpaca --data_dir ./Datasets/Alpaca/data --model llama3_2 --num_params 1B --finetune --run_type multi_gpu --use_lora --use_actv_ckpt --lr 1e-5 --data_type bf16 --load_weights
```

---

## 🧠 Model Sizes Supported

| Model    | Sizes Supported        |
| -------- | ---------------------- |
| GPT2     | 124M, 355M, 774M, 1.5B |
| LLaMA2   | 7B                     |
| LLaMA3   | 8B                     |
| LLaMA3.1 | 8B                     |
| LLaMA3.2 | 1B                     |

---

## 🛠️ Key Training Flags & Arguments

| Argument            | Description                               |
| ------------------- | ----------------------------------------- |
| `--model`           | Model name (GPT2, llama2, llama3)         |
| `--num_params`      | Model size (e.g., 124M, 7B)               |
| `--data_type`       | Data type: fp32, fp16, bf16               |
| `--load_weights`    | Load pretrained weights from Hugging Face |
| `--n_epochs`        | Number of training epochs                 |
| `--batch_size`      | Batch size                                |
| `--run_type`        | single\_gpu (default) or multi\_gpu       |
| `--lr`              | Learning rate (post-warmup)               |
| `--warmup_steps`    | Steps for learning rate warmup            |
| `--dataset`         | Dataset type: alpaca or Gutenberg         |
| `--finetune`        | Enable instruction finetuning             |
| `--use_zero_opt`    | Enable Zero optimizer                     |
| `--use_lora`        | Use LoRA for finetuning                   |
| `--lora_rank`       | Rank used in LoRA layers                  |
| `--lora_alpha`      | Scaling factor in LoRA                    |
| `--use_fsdp`        | Enable FSDP (multi-GPU)                   |
| `--mixed_precision` | Enable mixed precision training           |
| `--use_actv_ckpt`   | Enable activation checkpointing           |

---

## 🧪 Advanced Features

### ⚡ Mixed Precision Training

Use `--mixed_precision` to speed up training and save memory. Ideal for NVIDIA Tensor Cores (A100, V100).

### 🔄 FSDP (Fully Sharded Data Parallel)

Enable with `--use_fsdp` for large models on multi-GPU setups. Reduces memory usage by sharding model states.

### 🧩 LoRA

Use `--use_lora` to activate LoRA—an efficient method for finetuning large models with fewer parameters and lower compute.

### 🧠 Activation Checkpointing

Enable using `--use_actv_ckpt`. This breaks the computational graph and re-computes activations during backprop to save GPU memory.

---

## ⚠️ Notes

* CUDA must be installed and correctly configured.
* Use `--load_weights` to initialize from Hugging Face checkpoints.
* Easily scales to large datasets and multi-GPU setups.

---

## 🤝 Contributing

We welcome community contributions. Fork the repo and submit issues or pull requests if you have ideas, suggestions, or fixes.

---

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for full details.
