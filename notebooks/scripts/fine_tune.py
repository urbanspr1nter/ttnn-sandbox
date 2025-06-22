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