
import torch
from scripts.preload_dataloaders import load_pickled_dataloader
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.train import train_model_simple, save_model_and_optimizer
import tiktoken
import json
import os
from pathlib import Path

max_iterations = -1 
base_directory = "/home/rngo/code/ttnn-sandbox"

tokenizer = tiktoken.get_encoding("gpt2")

model_directory = Path(f"{base_directory}/notebooks/models")
if not os.path.exists(model_directory):
  os.mkdir(model_directory)

train_loader = load_pickled_dataloader(f"{base_directory}/notebooks/data/fineweb-10b/train_loader.dl")
print(f"Loaded train_loader. This loader contains {len(train_loader)} total batches.")

val_loader = load_pickled_dataloader(f"{base_directory}/notebooks/data/fineweb-10b/val_loader.dl")
print(f"Loaded val_loader. This loader contains {len(val_loader)} total batches.")

GPT_CONFIG_355M = {
  "vocab_size": 50257,   # Vocabulary size
  "context_length": 1024, # Context length
  "emb_dim": 1024,        # Embedding dimension (larger than 124M)
  "n_heads": 16,         # Number of attention heads (larger than 124M)
  "n_layers": 24,        # Number of layers (larger than 124M)
  "drop_rate": 0.0,      # Dropout rate
  "qkv_bias": False      # Query-key-value bias
}

model = GPTModel(GPT_CONFIG_355M)
model = torch.compile(model)
model = model.to("cuda").to(torch.bfloat16)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, fused=True)

# We have lots of data, so we can just train for a single epoch.
num_epochs = 1

# Set the model to training mode
model.train()

timer = PerfTimer()
timer.start()
train_losses, val_losses = train_model_simple(
  model=model,
  train_loader=train_loader,
  val_loader=val_loader,
  optimizer=optimizer,
  num_epochs=num_epochs,
  eval_freq=1000,
  eval_iter=100, # eval less frequently
  start_context="Every effort moves you",
  tokenizer=tokenizer,
  device="cuda",
  max_iter=max_iterations
)
timer.stop()

print(f"Took this long to train: {timer.elapsed_ms()} ms")

save_model_and_optimizer(
  model_path=f"{str(model_directory)}/gpt2-355M-model-bs32.pth",
  model=model,
  optimizer_path=f"{str(model_directory)}/optimizer-gpt2-355M-model-bs32.pth",
  optimizer=optimizer
)

with open(f"{str(model_directory)}/losses.json", "w") as f:
  f.write(
    json.dumps({
      "train_losses": train_losses,
      "val_losses": val_losses
    }, indent=2)
  )

print(f"🎉 Model has been trained!")