
import torch
from scripts.preload_dataloaders import load_train_dataloader, load_val_dataloader
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.train import train_model_simple
from scripts.generate import generate_text_simple
import tiktoken

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

train_loader = load_train_dataloader("/home/rngo/code/ttnn-sandbox/notebooks/data/fineweb-3b/train_loader.dl")
print("Loaded train_loader.")

val_loader = load_val_dataloader("/home/rngo/code/ttnn-sandbox/notebooks/data/fineweb-3b/val_loader.dl")
print("Loaded val_loader")

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
model = model.to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# We have lots of data, so we can just train for a single epoch.
num_epochs = 1

timer = PerfTimer()
timer.start()
train_losses, val_losses = train_model_simple(
    model, train_loader, val_loader, optimizer,
    num_epochs=num_epochs, eval_freq=50, eval_iter=50, # eval less frequently
    start_context="Every effort moves you", tokenizer=tokenizer, device="cuda"
)
timer.stop()

print(f"Took this long to train: {timer.elapsed_ms()} ms")

torch.save(model.state_dict(), "/home/rngo/code/ttnn-sandbox/notebooks/models/gpt2-355M-model.pth")

model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(
  torch.load("/home/rngo/code/ttnn-sandbox/notebooks/models/gpt2-355M-model.pth", weights_only=True)
)

perf_timer = PerfTimer()
perf_timer.start()
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=50,
    context_size=GPT_CONFIG_355M["context_length"]
)
perf_timer.stop()

print("Generated tokens in", perf_timer.elapsed_ms(), "ms")
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
