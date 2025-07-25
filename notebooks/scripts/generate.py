import torch
from scripts.util import token_ids_to_text
import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]

    with torch.no_grad():
      logits = model(idx_cond)
    
    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    
    idx = torch.cat((idx, idx_next), dim=1)

  return idx


def generate(
  model,
  idx,
  max_new_tokens,
  context_size,
  temperature=0.0,
  top_k=None,
  eos_id=None,
  device="cpu",
  stop_sequence=None,
  stream=False
):
    idx = idx.to(device)
    
    # Initialize tokenizer once if streaming
    if stream:
        tokenizer = tiktoken.get_encoding("gpt2")
    
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        if stream:
          idx_next_text = token_ids_to_text(idx_next, tokenizer)
          if idx_next_text != "\n\n":
            print(idx_next_text, end="", flush=True)

        if stop_sequence is not None:
          last_one = idx_next[0, -1:].tolist()
          if last_one == stop_sequence:
             break

    return idx
