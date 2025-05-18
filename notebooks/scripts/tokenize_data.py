from .perf_timer import PerfTimer
import pickle
import tiktoken

def tokenize(file_name, output_file):
  print(f"Tokenizing {file_name}")

  timer = PerfTimer()

  timer.start()
  tokenizer = tiktoken.get_encoding("gpt2")

  tokens = []
  with open(file_name, "r") as f:
    i = 0
    curr_line = f.readline()

    while curr_line:
      tokens.extend(
        tokenizer.encode(
          curr_line,
          allowed_special={"<|endoftext|>"}
        )
      )

      i += 1
      if i % 10000 == 0:
        print(f"Lines Read: {i}")

      curr_line = f.readline()

  with open(output_file, "wb") as f:
    pickle.dump(tokens, f)

  timer.stop()

  print(f"Lines Read: {i}")
  print(f"Tokenized {file_name}. Number of tokens: {len(tokens)}. Total lines: {i}. Time: {timer.elapsed_ms()}ms")

  return tokens

"""
Example usage in code:

tokenize(
  "/home/avgdev/code/ttnn-sandbox/notebooks/data/fineweb-3b/val_data.txt", 
  "val_tokens.txt"
) 
"""
