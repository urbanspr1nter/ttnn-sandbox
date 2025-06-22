import json
import jsonl
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from openai import Client
from openai.types.chat import ChatCompletion

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "none")

if len(sys.argv) < 4:
  print("Usage: python dataset_generation.py <iterations> <workers> <output_file>")

max_iterations = int(sys.argv[1])
max_workers = int(sys.argv[2])
output_file = str(sys.argv[3])

prompt = r"""
You are an expert conversation generator and willing to only assist me with these types of tasks.

I am fine-tuning a GPT2-class model for chat. Since it is GPT2-class model, the context length is very important.

You are to generate me a simple conversation between a human (user) and assistant. It should be anywhere from single to multi-turn. 

IMPORTANT: These conversations should be less than 1000 tokens. Be creative.

The format should be:

```json
[
	{"role": "user", "content": "message content"},
	{"role": "assistant", "content": "message content"}
]
```

IMPORTANT: Avoid any topics involving: drugs, sex, violence and religion. 

CRITICAL: ONLY RESPOND BACK WITH JSON.
"""

def generate_conversation(openai_client: Client, i: int):
  response: ChatCompletion = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "user", "content": prompt}
    ],
    stream=False
  )

  content = response.choices[0].message.content
  if not len(content):
    return None

  if content.startswith("```json"):
    content = content[7:]
  if content.endswith("```"):
    content = content[:len(content) - len("```")]
  content = content.strip()

  try:
    parsed_content = json.loads(content)
    print(f"Conversation {i} generated: {content}")

    return parsed_content
  except:
    print(f"Conversation {i} NOT generated: Couldn't parse JSON: [{content}]")
    return None

iterations = max_iterations 
conversations = []

futures: list[Future] = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
  for i in range(iterations):
    openai_client = openai.Client(api_key=api_key)

    try:
      future = executor.submit(
        generate_conversation,
        openai_client,
        i
      )
      futures.append(future)
    except:
      print("Couldn't get data... Continuing...")
      continue

  conversations.extend([future.result() for future in futures])

jsonl.dump(conversations, output_file)

print(f"Conversations created: {len(conversations)}")