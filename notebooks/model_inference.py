import os

import tiktoken
import torch

from scripts.generate import generate
from scripts.gpt2_common import GPT_CONFIG_355M
from scripts.gpt2_model import GPTModel
from scripts.model_loader import load_model_from_path
from scripts.perf_timer import PerfTimer
from scripts.util import text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")

base_directory = "/home/rngo/code/ttnn-sandbox"
model_directory = f"{base_directory}/notebooks/models"

model_name = input("Input the model filename found in the `models` directory. ")
if not model_name:
    print("Specify a model name.")
    exit(1)

model_file = f"{model_directory}/{model_name}"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:1"

if not os.path.exists(model_file):
    print(f"Could not find model at path {model_directory}/{model_file}")
    exit(1)

def load_quantized_model(model_path):
    """Load INT8 quantized model (CPU only)"""
    # Create model structure first
    model = GPTModel(GPT_CONFIG_355M)

    # Apply quantization structure (important for loading quantized weights)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Load the quantized state dict (always on CPU)
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove "_orig_mod." prefix if present (from torch.compile)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()

    return model


# Check if file is quantized or regular model
is_quantized = "int8" in model_file.lower() or "quantized" in model_file.lower()
if is_quantized:
    device = "cpu"

model = load_model_from_path(model_file, device=device)

step = 0
while step <= 10:

    perf_timer = PerfTimer()
    perf_timer.start()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("I like to play", tokenizer),
        max_new_tokens=100,
        context_size=1024,
        temperature=1.0,
        top_k=25,
        device=device,  # Use appropriate device
    )
    perf_timer.stop()

    print(
        f"For {step} - Generated tokens in",
        perf_timer.elapsed_ms(),
        f"ms (on {device})",
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    print()
    print()
    step += 1
