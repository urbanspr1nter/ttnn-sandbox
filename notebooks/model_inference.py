import torch
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.generate import generate_text_simple
from scripts.util import text_to_token_ids, token_ids_to_text
import tiktoken

base_directory = "/home/rngo/code/ttnn-sandbox"

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(
  torch.load(f"{base_directory}/notebooks/models/gpt2-355M-model.pth", weights_only=True)
)

model.eval()

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