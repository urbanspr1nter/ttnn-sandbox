import ttnn
import torch

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


def softmax_ttnn(x_ttnn):
  return ttnn.div(
    ttnn.exp(x_ttnn),
    ttnn.sum(
      ttnn.exp(x_ttnn),
      dim=-1
    )
  )