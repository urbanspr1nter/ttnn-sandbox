from .perf_timer import PerfTimer
import tiktoken

def flush(file_name, tokens):
  """
  We will need to flush tokens to a file once in a while
  """
  with open(file_name, "a") as f:
    f.write("\n".join([str(token) for token in tokens]))

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

        flush(output_file, tokens)
        tokens = []

      curr_line = f.readline()


  timer.stop()

  print(f"Tokenized {file_name}. Number of tokens: {len(tokens)}. Total lines: {i}. Time: {timer.elapsed_ms()}ms")

  return tokens

"""
Example usage in code:

tokenize(
  "/home/avgdev/code/ttnn-sandbox/notebooks/data/fineweb-3b/val_data.txt", 
  "val_tokens.txt"
) 
"""
