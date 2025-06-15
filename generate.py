import torch


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int = None,
    eos_id: int = None
) -> torch.Tensor:
    """
    Generate tokens using autoregressive decoding from a given prompt.

    Args:
        model (torch.nn.Module): The autoregressive model (e.g., GPT or LLaMA).
        idx (torch.Tensor): Tensor of token indices with shape (batch_size, current_seq_len).
        max_new_tokens (int): How many new tokens to generate.
        context_size (int): Maximum number of tokens model can see (context window).
        temperature (float): >0 adds randomness via softmax scaling; 0 for greedy decoding.
        top_k (int, optional): Keep only top-k highest probability tokens (sampling mode).
        eos_id (int, optional): If specified, generation stops when this token appears.

    Returns:
        torch.Tensor: Final generated token indices of shape (batch_size, input_len + new_tokens).
    """

    model.eval()  # Ensure model is in evaluation mode
    device = next(model.parameters()).device  # Get model device (e.g., cuda:0)

    # Ensure input is on the same device as the model
    idx = idx.to(device)

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only keep the most recent tokens for context (sliding window)
            idx_cond = idx[:, -context_size:]

            # Forward pass: get logits from model
            logits = model(idx_cond)

            # Get logits from the last timestep only (shape: [B, vocab_size])
            logits = logits[:, -1, :]

            # Apply top-k filtering (optional)
            if top_k is not None:
                # Get top-k logits and compute cutoff
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float('-inf')),
                    logits
                )

            # Sampling mode (stochastic decoding)
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding: pick the highest logit
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Early stopping: if eos_id is generated (batch-aware)
            if eos_id is not None:
                if (idx_next == eos_id).all():
                    break

            # Append the predicted token to the input tensor
            idx = torch.cat((idx, idx_next), dim=1)

    return idx
