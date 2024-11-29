import torch
from torch.utils.data import Dataset
from utils import tokenize, pad_sequence, create_vocab, calculate_max_len

class SMILESDataset(Dataset):
    def __init__(self, filepath, vocab=None, max_len=None):
        # Load data from csv file
        self.data = self.load_data(filepath)
        self.vocab = create_vocab(filepath) if vocab is None else vocab
        self.max_len = calculate_max_len(self.data) if max_len is None else max_len

    def load_data(self, filepath):
        # Load the data from the CSV file
        data = []
        with open(filepath, "r") as f:
            for line in f:
                reaction = line.strip()  # SMILES reaction
                data.append(reaction)
        return data

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get SMILES string for the reaction at index idx
        reaction = self.data[idx]
        
        # Tokenize the SMILES string using vocab
        tokens = tokenize(reaction, self.vocab)
        
        # Pad the sequence to ensure fixed length
        input_ids = pad_sequence(tokens, self.max_len, self.vocab["<pad>"])
        
        # Create attention mask (1 for non-padding tokens, 0 for padding tokens)
        attention_mask = [1 if token != self.vocab["<pad>"] else 0 for token in input_ids]
        
        # Return input_ids and attention_mask for model input
        return torch.tensor(input_ids), torch.tensor(attention_mask)

