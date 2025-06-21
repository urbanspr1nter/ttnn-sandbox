import torch
import json

GPT_CONFIG_355M = {
  "vocab_size": 50257,   # Vocabulary size
  "context_length": 1024, # Context length
  "emb_dim": 1024,        # Embedding dimension (larger than 124M)
  "n_heads": 16,         # Number of attention heads (larger than 124M)
  "n_layers": 24,        # Number of layers (larger than 124M)
  "drop_rate": 0.0,      # Dropout rate
  "qkv_bias": False      # Query-key-value bias
}

def save_model_and_optimizer(model_path: str, model, optimizer_path: str, optimizer):
  torch.save(model.state_dict(), model_path)
  torch.save(optimizer.state_dict(), optimizer_path)


def save_training_state(
    base_model_path: str,
    model: any,
    model_file_name: str,
    optimizer: any,
    optimizer_file_name: str,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    train_losses: list[float] = [],
    val_losses: list[float] = []
):
  state = {
    "epoch": epoch,
    "global_step": global_step,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_losses": train_losses,
    "val_losses": val_losses
  }

  with open(
    f"{base_model_path}/checkpoint_state.json", 
    "w", 
    encoding="utf-8"
  ) as f:
    f.write(json.dumps(state, indent=2))

  save_model_and_optimizer(
    model_path=f"{base_model_path}/{model_file_name}",
    model=model,
    optimizer_path=f"{base_model_path}/{optimizer_file_name}",
    optimizer=optimizer
  )