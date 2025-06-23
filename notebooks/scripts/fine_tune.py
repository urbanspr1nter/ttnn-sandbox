import torch
from torch.utils.data import Dataset

def format_input(entry):
  """
  Formats an entry in the JSON format: {role: "role", content: "message content"}
  to Alpaca style format.

  \n\n is injected at each response for special marking
  """
  instruction_text = (
    f"Below is an instruction that describes a task. "
    f"Write a response that appropriately completes the request."
  )

  for i, message in enumerate(entry):
    if i + 1 == len(entry) and message["role"] != "user":
      break

    if message["role"] == "user":
      instruction_text += f"\n\n### Instruction:\n{message['content']}"
    else:
      instruction_text += f"\n\n### Response:\n{message['content']}\n\n"

  # Add the prompt for the next response
  instruction_text += f"\n\n### Response:\n"

  return instruction_text

class InstructionDataset(Dataset):
  def __init__(self, data, tokenizer):
    self.data = data

    self.encoded_texts = []
    self.masks = []

    for item in self.data:
      tokens = []
      mask = []

      _instruction_base = f"Below is an instruction that describes a task. Write a response that appropriately completes the request"
      _instruction_base_tokens = tokenizer.encode(_instruction_base, allowed_special={"<|endoftext|>"})
      tokens.extend(_instruction_base_tokens)
      mask.extend([True] * len(_instruction_base_tokens))

      for message in item:
        if message["role"] == "user":
          _instruction_content_header = f"\n\n### Instruction:\n"
          _instruction_content_header_tokens = tokenizer.encode(_instruction_content_header, allowed_special={"<|endoftext|>"})
          tokens.extend(_instruction_content_header_tokens)
          mask.extend([True] * len(_instruction_content_header_tokens))

          _instruction_content_content = message["content"]
          _instruction_content_content_tokens = tokenizer.encode(_instruction_content_content, allowed_special={"<|endoftext|>"})
          tokens.extend(_instruction_content_content_tokens)
          mask.extend([True] * len(_instruction_content_content_tokens))
        else:
          _instruction_response_header = f"\n\n### Response:\n"
          _instruction_response_header_tokens = tokenizer.encode(_instruction_response_header, allowed_special={"<|endoftext|>"})
          tokens.extend(_instruction_response_header_tokens)
          mask.extend([True] * len(_instruction_response_header_tokens))

          # Dont mask the assistant response
          _instruction_response_content = message["content"] + "\n\n"
          _instruction_response_content_tokens = tokenizer.encode(_instruction_response_content, allowed_special={"<|endoftext|>"})
          tokens.extend(_instruction_response_content_tokens)
          mask.extend([False] * len(_instruction_response_content_tokens))

      self.encoded_texts.append(tokens)
      self.masks.append(mask)

      
  def __getitem__(self, index):
    return self.encoded_texts[index], self.masks[index]
  
  def __len__(self):
    return len(self.data)


def custom_collate_fn(
   batch,
   pad_token_id=50256,
   ignore_index=-100,
   allowed_max_length=None,
   device="cpu" 
):
  tokens_batch = [item[0] for item in batch]
  masks_batch = [item[1] for item in batch]

  # find the longest equence in the batch
  batch_max_length = max(len(tokens) + 1 for tokens in tokens_batch)

  inputs_lst, targets_lst = [], []

  for tokens, mask in zip(tokens_batch, masks_batch):
    new_tokens = tokens.copy()
    new_tokens += [pad_token_id]
    padded_tokens = (
      new_tokens + ([pad_token_id] * (batch_max_length - len(new_tokens)))
    )

    new_mask = mask.copy()
    new_mask += [True] # mask the added padded token
    padded_mask = (
      new_mask + ([True] * (batch_max_length - len(new_mask)))
    )

    inputs = torch.tensor(padded_tokens[:-1])
    targets = torch.tensor(padded_tokens[1:])

    pad_mask = targets == pad_token_id
    indices = torch.nonzero(pad_mask).squeeze()

    if indices.numel() > 1:
      targets[indices[1:]] = ignore_index

    # add the mask
    for j in range(min(len(padded_mask), len(targets))):
      if j + 1 < len(padded_mask) and padded_mask[j + 1]:
        targets[j] = ignore_index

    if allowed_max_length is not None:
      inputs = inputs[:allowed_max_length]
      targets = targets[:allowed_max_length]

    inputs_lst.append(inputs)
    targets_lst.append(targets)

  inputs_tensor = torch.stack(inputs_lst).to(device)
  targets_tensor = torch.stack(targets_lst).to(device)

  return inputs_tensor, targets_tensor

