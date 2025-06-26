import ttnn
import torch
import tiktoken

from scripts.util import token_ids_to_text, text_to_token_ids

def argmax_and_concat_ttnn(x_ttnn, probas_ttnn, device=None):
  """
  Unfortunately this is super inefficient. I need to figure out how
  we can offload all this to ttnn. Right now something is up with concat.
  """
  
  probas_ttnn = ttnn.to_layout(probas_ttnn, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
  probas_ttnn = ttnn.reshape(probas_ttnn, (1, 1, probas_ttnn.shape[0], probas_ttnn.shape[1]))
  
  max_arg_idx = ttnn.argmax(
    probas_ttnn,
    dim=-1
  )
  # Ensure max_arg_idx has the same dtype as x_ttnn
  max_arg_idx = ttnn.reshape(max_arg_idx, (1, 1))

  result_torch = torch.concat((
    ttnn.to_torch(x_ttnn),
    ttnn.to_torch(max_arg_idx)
  ), dim=-1)

  result_ttnn = ttnn.from_torch(result_torch,dtype=ttnn.uint32, device=device)
  
  return result_ttnn

def generate_text_simple_ttnn(model, idx_ttnn, max_new_tokens, context_size, device):
  for _ in range(max_new_tokens):
    if idx_ttnn.shape[1] < context_size:
      idx_cond = idx_ttnn[:, :idx_ttnn.shape[1]]
    else:
      idx_cond = idx_ttnn[:, -context_size:]

    logits_ttnn = model(idx_cond)
    logits_ttnn = logits_ttnn[:, -1, :]
    probas_ttnn = softmax_ttnn(logits_ttnn)

    # this is still being done on CPU during concat
    idx_ttnn = argmax_and_concat_ttnn(idx_ttnn, probas_ttnn, device=device)

  return idx_ttnn

def generate_ttnn(
  model,
  idx,
  max_new_tokens,
  context_size,
  temperature=0.0,
  top_k=None,
  eos_id=None,
  device=None,
  stop_sequence=None,
  stream=False
):
  if device is None:
    print("Must provide a device")
    return

  if stream:
    tokenizer = tiktoken.get_encoding("gpt2")

  inf_tensor = torch.tensor(float("-inf"))

  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:] 
  
    idx_cond_ttnn = ttnn.from_torch(
      idx_cond,
      dtype=ttnn.uint32,
      layout=ttnn.TILE_LAYOUT,
      device=device
    )

    with torch.no_grad():
      logits_ttnn = model(idx_cond_ttnn)
    logits_ttnn = logits_ttnn[:, -1, :]

    ## Transfer everything back to torch to make life easier:
    logits = ttnn.to_torch(logits_ttnn)

    if top_k is not None:
      top_logits, _ = torch.topk(logits, top_k)
      min_val = top_logits[:, -1]
      logits = torch.where(logits < min_val, inf_tensor, logits)
    
    if temperature > 0.0:
      probs = torch.softmax(logits / temperature, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
    else:
      idx_next = torch.argmax(logits, dim=-1, keepdim=True)
 
    if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
      break

    idx = torch.cat((idx, idx_next), dim=1)
    
    if stream:
      idx_next_text = token_ids_to_text(idx_next, tokenizer)
      if idx_next_text != "\n\n":
        print(idx_next_text, end="", flush=True)

    if stop_sequence is not None:
      last_one = idx_next[0, -1:].tolist()
      if last_one == stop_sequence:
          break

  return idx

def softmax_ttnn(x_ttnn):
  return ttnn.div(
    ttnn.exp(x_ttnn),
    ttnn.sum(
      ttnn.exp(x_ttnn),
      dim=-1
    )
  )