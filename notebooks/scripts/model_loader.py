from scripts.gpt2_model import GPTModel
from scripts.gpt2_common import GPT_CONFIG_355M
import torch

def load_model_from_path(path, device, model_config_overrides=None):
  gpt2_config = GPT_CONFIG_355M.copy()  # Create a copy to avoid modifying the original
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
  
  # Move to device BEFORE compiling
  if device.startswith("cuda"):
    model = model.to(device).to(torch.bfloat16)
  else:
    model = model.to(device)
  
  # Compile AFTER moving to device
  model = torch.compile(model)

  return model
