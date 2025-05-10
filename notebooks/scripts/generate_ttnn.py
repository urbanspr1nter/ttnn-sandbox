import ttnn
import torch
import tiktoken
from torch import nn

def generate_text_simple_ttnn(model, idx_ttnn, max_new_tokens, context_size, device):
  """
  Note: this could be more efficient. im moving things around waaay too much here between
  CPU and TPU.
  """
  idx = ttnn.to_torch(idx_ttnn, device=device)
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]

    with torch.no_grad():
      logits_ttnn = model(ttnn.from_torch(idx_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
      logits = ttnn.to_torch(logits_ttnn, device=device)

    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    
    idx = torch.cat((idx, idx_next), dim=1)

  return idx