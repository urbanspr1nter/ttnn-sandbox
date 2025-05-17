import torch

def generator_load_token_data(file_name):
  with open(file_name, "r") as f:
    for l in f:
      yield int(l)

def load_token_data(file_name):
  tokens = []

  i = 0
  for line in generator_load_token_data(file_name):
    token_id = line
    tokens.append(token_id)

    i += 1

    if i % 10000 == 0:
      print(f"Processed token {i}")
  
  return torch.tensor(tokens)

def load_tokens(file_name):
  return torch.load(file_name)

def save_tensor_from_tokens(torch_tensor, file_name):
  torch.save(torch_tensor, file_name)

"""
t = load_token_data("/home/avgdev/code/ttnn-sandbox/notebooks/data/fineweb-3b/val_tokens.txt")
save_tensor_from_tokens(t, "val_tokens.pth")

print(t)
print(t.shape)
"""
