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

def create_train_dataloader(cfg, pickled_file_name, pickled_train_loader_file_name):
  with open(pickled_file_name, "rb") as f:
    train_tokens = pickle.load(f)

  with open(pickled_train_loader_file_name, "wb") as f:
    train_loader = create_dataloader_v2(
      train_tokens,
      batch_size=1,
      max_length=cfg["context_length"],
      stride=cfg["context_length"],
      drop_last=True,
      shuffle=True,
      num_workers=0
    )

    pickle.dump(train_loader, f)


def create_val_dataloader(cfg, pickled_file_name, pickled_val_loader_file_name):
  with open(pickled_file_name, "rb") as f:
    val_tokens = pickle.load(f)

  with open(pickled_val_loader_file_name, "wb") as f:
    val_loader = create_dataloader_v2(
      val_tokens,
      batch_size=1,
      max_length=cfg["context_length"],
      stride=cfg["context_length"],
      drop_last=True,
      shuffle=True,
      num_workers=0
    )

    pickle.dump(val_loader, f)


def load_train_dataloader(pickled_train_loader_file_name):
  with open(pickled_train_loader_file_name, "rb") as f:
    train_loader = pickle.load(f)

  return train_loader

def load_val_dataloader(pickled_val_loader_file_name):
  with open(pickled_val_loader_file_name, "rb") as f:
    val_loader = pickle.load(f)

  return val_loader

