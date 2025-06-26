import json
import os

import tiktoken
import torch

from scripts.fine_tune import format_input
from scripts.generate import generate
from scripts.generate_ttnn import generate_ttnn
from scripts.gpt2_common import GPT_CONFIG_355M
from scripts.load_pretrained_model_ttnn import load_pretrained_gpt2_model_ttnn
from scripts.model_loader import load_model_from_path
from scripts.util import token_ids_to_text, text_to_token_ids

# Configure TTNN related things
os.environ["TTNN_CONFIG_OVERRIDES"] = f'{{"enable_fast_runtime_mode": true, "enable_model_cache": true, "enable_logging": false}}'

def get_model_path():
  path_model_very_first = "models/demo/bare.pth"
  path_model_pre_trained = "models/demo/pretrained.pth"
  path_model_it_no_stop = "models/demo/it-no-stop.pth"
  path_model_it_completed = "models/gpt2-355M-model-it-scheduled-lr.pth"

  print("Which would you like to load?")
  print("1. Raw untrained model.")
  print("2. Pretrained, base model.")
  print("3. Instruction-tuned model with no stop sequences.")
  print("4. Final, complete model.")
  print("5. Custom Model")

  choice = int(input("Your choice: "))
  if choice < 1 or choice > 5:
    print("Invalid option. Defaulting to (1).")
    choice = 1

  if choice == 1:
    model_path = path_model_very_first
  elif choice == 2:
    model_path = path_model_pre_trained
  elif choice == 3:
    model_path = path_model_it_no_stop
  elif choice == 4:
    model_path = path_model_it_completed
  elif choice == 5:
    model_path = input("Your custom path: ")

  if not os.path.exists(model_path):
    print("Invalid path!")
    exit(1)

  return model_path

def get_device():
  print("Device?")
  print("1. CPU")
  if torch.cuda.is_available():
    print("2. CUDA")

  # Try to query fro tenstorrent hardware
  try:
    import ttnn
    if len(ttnn.get_device_ids()):
      print("9. Tenstorrent")
  except:
    pass

  device_choice = int(input("Input your device choice: "))

  device = "cpu"
  if device_choice == 2:
    device = "cuda"
  elif device_choice == 9:
    device = "tt"

  return device

def format_messages_to_tokens(messages, tokenizer, device="cpu"):
  formatted = format_input(messages)
  tokens = text_to_token_ids(formatted, tokenizer)

  if device == "tt":
    return tokens

  tokens = tokens.to(device)

  return tokens

def send_message(
  messages,
  device="cpu",
  tt_device=None,
  stream=True,
  stop_sequence=None
):
  start_context = format_messages_to_tokens(messages, tokenizer, device)

  if tt_device is not None:
    token_ids = generate_ttnn(
      model,
      idx=start_context,
      max_new_tokens=512,
      context_size=1024,
      temperature=0.6,
      top_k=20,
      eos_id=50256,
      device=tt_device,
      stop_sequence=stop_sequence,
      stream=stream
    )
  else:
    token_ids = generate(
      model,
      idx=start_context,
      max_new_tokens=512,
      context_size=1024,
      temperature=0.6,
      top_k=20,
      eos_id=50256,
      device=device,
      stop_sequence=stop_sequence,
      stream=stream
    )

  response = token_ids_to_text(token_ids, tokenizer)
  start_idx = response.rfind("\n\n### Response:\n")
  assistant_message = response[start_idx + len("\n\n### Response\n"):].strip()

  return assistant_message

tokenizer = tiktoken.get_encoding("gpt2")

double_newline_id = tokenizer.encode('\n\n', allowed_special={"<|endoftext|>"})[0]
stop_sequence = [double_newline_id]

model_path = get_model_path()
device = get_device()

tt_device = None
if device == "tt":
  
  import ttnn
  
  tt_device = ttnn.open_device(device_id=0)

  # Speeds things up a bit
  tt_device.enable_program_cache()

  model = load_pretrained_gpt2_model_ttnn(
    model_path,
    GPT_CONFIG_355M,
    device=tt_device
  )

  # Tenstorrent hardware normally needs a warmup messasge to perform reasonably well
  send_message(
    messages=[{"role": "user", "content": "Hello!"}],
    device=device,
    tt_device=tt_device,
    stream=False,
    stop_sequence=stop_sequence
  )
else:
  model = load_model_from_path(
    model_path,
    device
  )
  model.eval()


user_message = ""
messages = []
while True:
  user_message = input("# ")

  if user_message == "/bye":
    break
  elif user_message == "/reset":
    print("Resetted context.")
    messages = []
    continue
  elif user_message == "/messages":
    print(json.dumps(messages, indent=2))
    continue
  elif user_message.startswith("/"):
    print("Invalid command.")
    continue

  messages.append(
    {
      "role": "user",
      "content": user_message
    }
  )

  print("> ", end="", flush=True)

  result = send_message(
    messages,
    device=device,
    tt_device=tt_device,
    stop_sequence=stop_sequence
  )

  print()

  messages.append({"role": "assistant", "content": result})

if tt_device:
  ttnn.close_device(tt_device)