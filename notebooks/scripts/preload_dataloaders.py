from .prepare_data import create_dataloader_v2
import pickle

GPT_CONFIG_355M = {
  "vocab_size": 50257,   # Vocabulary size
  "context_length": 1024, # Context length
  "emb_dim": 1024,        # Embedding dimension (larger than 124M)
  "n_heads": 16,         # Number of attention heads (larger than 124M)
  "n_layers": 24,        # Number of layers (larger than 124M)
  "drop_rate": 0.0,      # Dropout rate
  "qkv_bias": False      # Query-key-value bias
}

def create_dataloader_to_pickle(cfg, pickled_tokens_file_name, pickled_loader_file_name, batch_size=1):
  with open(pickled_tokens_file_name, "rb") as f:
    tokens = pickle.load(f)

  with open(pickled_loader_file_name, "wb") as f:
    dl = create_dataloader_v2(
      tokens,
      batch_size=batch_size,
      max_length=cfg["context_length"],
      stride=cfg["context_length"],
      drop_last=True,
      shuffle=True,
      num_workers=0
    )

    pickle.dump(dl, f)

def load_pickled_dataloader(pickled_loader_file_name):
  with open(pickled_loader_file_name, "rb") as f:
    dl = pickle.load(f)

  return dl

