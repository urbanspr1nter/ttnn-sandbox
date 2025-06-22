"""
This script will transform all messages in the conversations dataset into Alpaca format and compute
the total tokens. If the tokens exceed 1000, they will be filtered away.
"""

import jsonl
import random
import sys
import tiktoken
from notebooks.scripts.fine_tune import format_input

if len(sys.argv) < 3:
  print("Usage: python dataset_filter.py <input_file> <output_file>")
  exit(1)

dataset_path = sys.argv[1]

max_length = -1

with open(dataset_path, "r", encoding="utf-8") as f:
  all_data = list(jsonl.load(f))

# We will want to shuffle the dataset for good practice in case it already isn't.
# This is to ensure situations where dataset has items like this 500, 500, 500, 500, 1000, 1000, etc.
# When it comes to number of tokens.
random.shuffle(all_data)

# Remove any Nones
all_data = list(filter(lambda x: x is not None, all_data))

tokenizer = tiktoken.get_encoding("gpt2")

final = []
for i, raw_conversation in enumerate(all_data):
  conversation = format_input(raw_conversation)

  tokens = tokenizer.encode(conversation, allowed_special={"<|endoftext|>"})

  print(f"Conversation {i} has context length: {len(tokens)}")

  if len(tokens) < 1000:
    print("-> Passed!")
    final.append(raw_conversation)

    # Fun statistic to keep track of the max token length we have seen and allowed. 
    # It should ideally be 999.
    if max_length < len(tokens):
      max_length = len(tokens)
  else:
    print("-> Failed!")
    print(len(tokens))

print(f"Passed token length: {len(final)}")
print(f"Failed token length: {len(all_data) - len(final)}")
print(f"Max in the new dataset: {max_length} tokens")

jsonl.dump(final, "conversations-filtered.jsonl")