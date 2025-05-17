import pickle 

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
  
  return tokens

def load_tokens(file_name):
  with open(file_name, "rb") as f:
    return pickle.load(f)

def save_tokens(tokens, file_name):
  with open(file_name, "wb") as f:
    pickle.dump(tokens, f)

"""
t = load_token_data("/home/avgdev/code/ttnn-sandbox/notebooks/data/fineweb-3b/train_tokens.txt")
save_tokens(t, "train_tokens.pth")

print(t)
print(t.shape)
"""