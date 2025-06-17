import torch
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.generate import generate_text_simple, generate
from scripts.model_loader import load_model_from_path
from scripts.util import text_to_token_ids, token_ids_to_text
import tiktoken

base_directory = "/home/rngo/code/ttnn-sandbox"

tokenizer = tiktoken.get_encoding("gpt2")

step = 180000
model = load_model_from_path(
  f"{base_directory}/notebooks/models/checkpoint-model-{step}.pth",
  "cpu"
)
model.eval()

perf_timer = PerfTimer()
perf_timer.start()
token_ids = generate(
    model=model,
    idx=text_to_token_ids("I like to play", tokenizer),
    max_new_tokens=50,
    context_size=1024,
    temperature=0.8,
    top_k=20
)
perf_timer.stop()

print("Generated tokens in", perf_timer.elapsed_ms(), "ms")
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))