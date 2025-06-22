import torch
from scripts.gpt2_common import GPT_CONFIG_355M
from scripts.gpt2_model import GPTModel

device = "cuda:0"
model = GPTModel(GPT_CONFIG_355M).to(device).to(torch.bfloat16)

torch.save(model.state_dict(), "bare.pth")