import torch
from scripts.gpt2_model import GPTModel
from scripts.perf_timer import PerfTimer
from scripts.generate import generate_text_simple, generate
from scripts.model_loader import load_model_from_path
from scripts.util import text_to_token_ids, token_ids_to_text
import tiktoken

base_directory = "/home/rngo/code/ttnn-sandbox"

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

# Configuration for GPT-2 355M
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": False
}

def load_quantized_model(model_path):
    """Load INT8 quantized model (CPU only)"""
    # Create model structure first
    model = GPTModel(GPT_CONFIG_355M)
    
    # Apply quantization structure (important for loading quantized weights)
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Load the quantized state dict (always on CPU)
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Remove "_orig_mod." prefix if present (from torch.compile)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

# Check if file is quantized or regular model
model_file = f"{base_directory}/notebooks/models/gpt2-355M-int8-quantized-230000.pth"
is_quantized = "int8" in model_file.lower() or "quantized" in model_file.lower()

step = 0
while step <= 10:
    if is_quantized:
        # Load quantized model (CPU only)
        print("Loading INT8 quantized model (CPU inference)...")
        model = load_quantized_model(model_file)
        device = "cpu"
    else:
        # Load regular model (can use CUDA)
        model = load_model_from_path(model_file, "cuda")
        device = "cuda"

    perf_timer = PerfTimer()
    perf_timer.start()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("I like to play", tokenizer),
        max_new_tokens=100,
        context_size=1024,
        temperature=1.0,
        top_k=25,
        device=device  # Use appropriate device
    )
    perf_timer.stop()

    print(f"For {step} - Generated tokens in", perf_timer.elapsed_ms(), f"ms (on {device})")
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    print()
    print()
    step += 1
