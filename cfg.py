import torch

class Config:
    # Data-related hyperparameters
    filepath = './data/data.csv'  # Path to the dataset (CSV file with reactions)
    batch_size = 32

    # Model-related hyperparameters
    latent_dim = 128  # Latent space dimension (adjust as needed)
    input_dim = 128  # Dimension of input data (e.g., SMILES length after padding)
    hidden_dim = 256  # Hidden layer dimension for both encoder and decoder
    num_heads = 4
    num_layers = 3
    dropout = 0.1
    
    # Training-related hyperparameters
    learning_rate = 1e-3
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beta = 0.1 # Weight for KL divergence loss
