from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class GPTDatasetV2(Dataset):
    def __init__(self, tokens, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        j = 0
        tokens_length = len(tokens)
        for i in range(0, tokens_length - max_length, stride):
            if j % 100000 == 0:
                print(f"Processing chunk: {j}. Token: {i} of {tokens_length}")

            input_chunk = torch.tensor(tokens[i:i + max_length])
            target_chunk = torch.tensor(tokens[i + 1:i + max_length + 1])
            
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

            j += 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            torch_input_chunk = torch.tensor(input_chunk)
            torch_target_chunk = torch.tensor(target_chunk)

            self.input_ids.append(torch_input_chunk)
            self.target_ids.append(torch_target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v2(tokens, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Create dataset
    dataset = GPTDatasetV2(tokens, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader