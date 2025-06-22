import torch
import os
from scripts.generate import generate
from scripts.util import token_ids_to_text, text_to_token_ids
from scripts.fine_tune import format_input
from scripts.model_loader import load_model_from_path
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

path_model_very_first = "models/demo/bare.pth"
path_model_pre_trained = "models/demo/pretrained.pth"
path_model_it_no_stop = "models/demo/it-no-stop.pth"
path_model_it_completed = "models/demo/complete.pth"

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

print("Device?")
print("1. CPU")
if torch.cuda.is_available():
  print("2. CUDA")

device_choice = int(input("Input your device choice: "))

device = "cpu"
if device_choice == 2:
  device = "cuda:0"

model = load_model_from_path(model_path, device)
model.eval()

double_newline_id = tokenizer.encode('\n\n', allowed_special={"<|endoftext|>"})[0]
stop_sequence = [double_newline_id]

def send_message(messages):
  token_ids = generate(
    model,
    idx=text_to_token_ids(format_input(messages), tokenizer).to(device),
    max_new_tokens=512,
    context_size=1024,
    temperature=0.6,
    top_k=20,
    eos_id=50256,
    device=device,
    stop_sequence=stop_sequence
  )
  response = token_ids_to_text(token_ids, tokenizer)
  start_idx = response.rfind("\n\n### Response:\n")
  assistant_message = response[start_idx + len("\n\n### Response\n"):].strip()

  return assistant_message

user_message = ""
while True:
  user_message = input("# ")

  if user_message == "/bye":
    break

  messages = [
    {
      "role": "user",
      "content": user_message
    }
  ]

  result = send_message(messages)

  print(f"> {result}")

  messages.append({"role": "assistant", "content": result})
