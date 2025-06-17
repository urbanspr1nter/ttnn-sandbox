import torch

def apply_temperature_scaling(temperature: float, logits):
  if temperature == 0.0:
    return torch.softmax(logits, dim=-1)
  else:
    logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    return probs
