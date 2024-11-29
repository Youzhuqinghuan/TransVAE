# utils.py

# Tokenize SMILES strings into a list of integers corresponding to vocab indices
def tokenize(smiles, vocab):
    return [vocab[char] for char in smiles]

# Pad a sequence of token IDs to a fixed length
def pad_sequence(tokens, max_len, pad_token_id):
    # If the sequence is shorter than max_len, pad it
    return tokens + [pad_token_id] * (max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]

# Create vocabulary from SMILES data
def create_vocab(filepath):
    chars = set()
    with open(filepath, "r") as f:
        for line in f:
            chars.update(line.strip())
    vocab = {char: idx for idx, char in enumerate(sorted(chars))}
    vocab["<pad>"] = len(vocab)  # Add padding token
    return vocab

def calculate_max_len(data):
        # Calculate the maximum length of SMILES strings in the dataset
        return max(len(line) for line in data)