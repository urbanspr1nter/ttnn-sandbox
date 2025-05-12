import ttnn
import torch
import tiktoken
from torch import nn

def manual_argmax_ttnn(
  tensor, dim=-1, keepdim=True, device=None
):
  """
  Super inefficient
  """
  torch_tensor = ttnn.to_torch(tensor)
  result = torch.argmax(
    torch_tensor,
    dim=dim,
    keepdim=keepdim
  )

  result_ttnn = ttnn.from_torch(
    result,
    dtype=ttnn.uint32,
    device=device
  )

  return result_ttnn

def manual_concat(ttnn_1, ttnn_2, dim=-1, device=None):
  """
  Super inefficient
  """

  result_torch = torch.cat((
    ttnn.to_torch(ttnn_1),
    ttnn.to_torch(ttnn_2)
  ), dim=dim)

  result_ttnn = ttnn.from_torch(
    result_torch,
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

    idx_next = manual_argmax_ttnn(
      probas_ttnn,
      dim=-1,
      keepdim=True,
      device=device
    )

    idx_ttnn = manual_concat(idx_ttnn, idx_next, dim=1, device=device)

  print(idx_ttnn)
  return idx_ttnn[0]


def softmax_ttnn(x_ttnn):
  return ttnn.div(
    ttnn.exp(x_ttnn),
    ttnn.sum(
      ttnn.exp(x_ttnn),
      dim=-1
    )
  )