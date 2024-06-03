import torch
from transformers import DataCollatorWithPadding

class SimpleDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, examples):
        # `DataCollatorWithPadding` handles the padding of input ids and attention mask.
        return super().__call__(examples)

def make_training_dataset(tokenizer, max_sequence_length=128, num_samples=1000):
    """
    Generate a simple dataset of sequences for training.
    
    Parameters:
    - tokenizer: the tokenizer to use for encoding text.
    - max_sequence_length: the maximum length of sequences.
    - num_samples: the number of samples to generate.

    Returns:
    - A PyTorch dataset containing tokenized input sequences.
    """
    # Simulate some basic text data: repetitive patterns to mimic structured data
    vocabulary_size = tokenizer.vocab_size
    data = []
    for _ in range(num_samples):
        length = torch.randint(low=10, high=max_sequence_length, size=(1,)).item()
        sequence = torch.randint(low=1, high=vocabulary_size, size=(length,))  # generate random token ids
        data.append(tokenizer.decode(sequence.tolist()))

    # Tokenize the generated data
    encoded_inputs = tokenizer(data, add_special_tokens=True, truncation=True, padding='max_length', max_length=max_sequence_length, return_tensors="pt")

    # Create a PyTorch dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    return SimpleDataset(encoded_inputs)