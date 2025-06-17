from scripts.gpt2_model import GPTModel
import torch

GPT_CONFIG_355M = {
  "vocab_size": 50257,   # Vocabulary size
  "context_length": 1024, # Context length
  "emb_dim": 1024,        # Embedding dimension (larger than 124M)
  "n_heads": 16,         # Number of attention heads (larger than 124M)
  "n_layers": 24,        # Number of layers (larger than 124M)
  "drop_rate": 0.0,      # Dropout rate
  "qkv_bias": False      # Query-key-value bias
}

def load_model_from_path(path, device, model_config_overrides=None):
  gpt2_config = GPT_CONFIG_355M
  if model_config_overrides is not None:
    gpt2_config.update(model_config_overrides)

  state_dict = torch.load(path, map_location=device)

  fixed_state_dict = {}
  for key, value in state_dict.items():
    cleaned_key = key
    if key.startswith('_orig_mod.'):
      cleaned_key = key[10:]
    elif key.startswith('module.'):
      cleaned_key = key[7:]
    elif key.startswith('module._orig_mod.'):
      cleaned_key = key[17:]

    fixed_state_dict[cleaned_key] = value

  model = GPTModel(gpt2_config)
  model.load_state_dict(fixed_state_dict)
  model = torch.compile(model)
  
  if device == "cuda":
    model.to(device).to(torch.bfloat16)
  else:
    model.to(device)

  return model
