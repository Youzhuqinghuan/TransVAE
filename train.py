import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import tokenize, pad_sequence, create_vocab, calculate_max_len
from dataset import SMILESDataset
from model import TransformerVAE
from cfg import Config

def train_epoch(model, dataloader, optimizer, criterion, device, beta):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        optimizer.zero_grad()
        logits, mu, logvar = model(input_ids, attention_mask)
        loss = criterion(logits, input_ids, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, beta):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            logits, mu, logvar = model(input_ids, attention_mask)
            loss = criterion(logits, input_ids, mu, logvar, beta)
            
            total_loss += loss.item()
    return total_loss / len(dataloader)

def vae_loss(logits, target, mu, logvar, beta):
    # Reconstruction loss
    recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='sum')
    
    # KL divergence loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_div

def main():
    # Load configuration
    config = Config()
    
    # Load dataset
    dataset = SMILESDataset(config.filepath)

    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model, optimizer, and criterion
    model = TransformerVAE(
        vocab_size=len(dataset.vocab),
        embedding_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = vae_loss
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoint_dir = f"./checkpoints/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, config.beta)
        val_loss = validate_epoch(model, val_loader, criterion, config.device, config.beta)
        
        print(f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
        
        # Save the model at the end of each epoch
        torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pth")
    
    # Save the final model
    torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pth")
    
    # Test the model
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", weights_only=True))
    test_loss = validate_epoch(model, test_loader, criterion, config.device, config.beta)
    print(f"Test Loss: {test_loss:.4f}")
    
if __name__ == "__main__":
    main()