
import torch
from scripts.preload_dataloaders import load_pickled_dataloader
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.train import train_model_simple
from scripts.gpt2_common import save_model_and_optimizer
from scripts.model_loader import load_model_from_path
import tiktoken
import json
import os
from pathlib import Path
import gc

gc.collect()

tokenizer = tiktoken.get_encoding("gpt2")

base_directory = "/home/rngo/code/ttnn-sandbox"
if not os.path.exists(base_directory):
  print("Invalid base directory specified.")
  exit(1)

model_directory = Path(f"{base_directory}/notebooks/models")
if not os.path.exists(model_directory):
  os.mkdir(model_directory)

device = 'cuda:1'
dataset = 'fineweb-1b'
max_iterations = -1

confirmation_max_iterations = input(f"We will run the training for {max_iterations} iterations. Confirm? [Y/n]");
if confirmation_max_iterations.lower() == "n":
  print("Exiting training...")
  exit(1)

# We have lots of data, so we can just train for a single epoch.
num_epochs = int(input("How many epochs? [Default: 1]"))
if num_epochs <= 0:
  print("Setting to the default of 1 epoch.")
  num_epochs = 1


train_loader = load_pickled_dataloader(f"{base_directory}/notebooks/data/{dataset}/train_loader.dl")
print(f"Loaded train_loader. This loader contains {len(train_loader)} total batches.")

val_loader = load_pickled_dataloader(f"{base_directory}/notebooks/data/{dataset}/val_loader.dl")
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

if device.startswith("cuda"):
  torch.cuda.empty_cache()

  capability = torch.cuda.get_device_capability()
  if capability[0] >= 7:
    print("Modern CUDA device found. Using tensor cores.")
    torch.set_float32_matmul_precision("high")
  else:
    print("Tensor cores not supported on this CPU. Will still proceed.")

  print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

#model = load_model_from_path(
#  f"{model_directory}/checkpoint-model-150000.pth",
#  device
#)


model = GPTModel(GPT_CONFIG_355M)

if device.startswith("cuda"):
  model = model.to(device).to(torch.bfloat16)
else:
  model = model.to(device)

# compile after moving the model.
model = torch.compile(model)

optimizer = torch.optim.AdamW(
  model.parameters(),
  lr=2e-4,
  weight_decay=0.1,
  fused=True
)


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
  device=device,
  max_iter=max_iterations
)
timer.stop()

print(f"Took this long to train: {timer.elapsed_ms()} ms")

save_model_and_optimizer(
  model_path=f"{str(model_directory)}/gpt2-355M-model-{dataset}.pth",
  model=model,
  optimizer_path=f"{str(model_directory)}/optimizer-gpt2-355M-model-{dataset}.pth",
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

gc.collect()
if device.startswith("cuda"):
  torch.cuda.empty_cache()