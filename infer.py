# inference.py
import torch
from model import TransformerVAE
from dataset import create_vocab
from config import Config
from utils import tokenize, pad_sequence

def infer(smiles):
    cfg = Config()
    vocab = create_vocab(cfg.data_path)
    model = TransformerVAE(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_heads, cfg.num_layers, cfg.dropout).to("cuda")
    model.load_state_dict(torch.load(cfg.save_model_path))
    model.eval()

    tokens = tokenize(smiles, vocab)
    input_ids = pad_sequence(tokens, cfg.max_len, vocab["<pad>"])
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda")

    with torch.no_grad():
        mu, logvar = model.encode(input_tensor)
        z = model.reparameterize(mu, logvar)
    return z.cpu().numpy()

if __name__ == "__main__":
    smiles = "CCO>>C=O"  # Example input
    latent_vector = infer(smiles)
    print(latent_vector)
