import torch
from scripts.generate import generate
from scripts.util import token_ids_to_text, text_to_token_ids
from scripts.fine_tune import format_input
from scripts.model_loader import load_model_from_path
import tiktoken

device = "cuda:1"
tokenizer = tiktoken.get_encoding("gpt2")
model = load_model_from_path(f"models/gpt2-355M-model-it-ep2.pth", device)
model.eval()

def send_message(messages):
  token_ids = generate(
    model,
    idx=text_to_token_ids(format_input(messages), tokenizer).to(device),
    max_new_tokens=200,
    context_size=1024,
    temperature=0.6,
    top_k=20,
    eos_id=50256,
    device=device
  )
  response = token_ids_to_text(token_ids, tokenizer)
  start_idx = response.rfind("### Response:\n")
  assistant_message = response[start_idx + len("### Response\n"):].strip()

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
