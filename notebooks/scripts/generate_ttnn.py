import ttnn
import torch

def argmax_and_concat(x_ttnn, probas_ttnn, device=None):
  """
  Unfortunately this is super inefficient. I need to figure out how
  we can offload all this to ttnn
  """

  x_torch = ttnn.to_torch(x_ttnn)
  probas_torch = ttnn.to_torch(probas_ttnn)

  max_arg_idx = torch.argmax(
    probas_torch,
    dim=-1,
    keepdim=True
  )
  
  result_torch = torch.cat((
    x_torch,
    max_arg_idx 
  ), dim=-1)

  result_ttnn = ttnn.from_torch(
    result_torch,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.uint32,
    device=device
  )

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

    # this is still being done on CPU
    idx_ttnn = argmax_and_concat(idx_ttnn, probas_ttnn, device=device)

  return idx_ttnn


def softmax_ttnn(x_ttnn):
  return ttnn.div(
    ttnn.exp(x_ttnn),
    ttnn.sum(
      ttnn.exp(x_ttnn),
      dim=-1
    )
  )