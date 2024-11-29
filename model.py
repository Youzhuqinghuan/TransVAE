import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_heads, num_layers, dropout):
        super(TransformerVAE, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )

        # Latent space
        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)
        self.fc_latent_to_embedding = nn.Linear(latent_dim, embedding_dim)

        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )

        # Output projection to vocab size
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def encode(self, x, attention_mask):
        """
        Encodes input sequences into latent representations.
        """
        x = self.embedding(x).permute(1, 0, 2)  # [batch, seq_len, embedding_dim] -> [seq_len, batch, embedding_dim]
        h = self.encoder(x, src_key_padding_mask=(attention_mask == 0)).mean(dim=0)  # Mean pooling over sequence
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, sigma).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x, attention_mask):
        """
        Decodes latent vectors z into output sequences.
        """
        z = self.fc_latent_to_embedding(z).unsqueeze(0)  # [batch, embedding_dim] -> [1, batch, embedding_dim]
        x = self.embedding(x).permute(1, 0, 2)  # [batch, seq_len, embedding_dim] -> [seq_len, batch, embedding_dim]
        output = self.decoder(x, z, tgt_key_padding_mask=(attention_mask == 0))
        return self.output_proj(output.permute(1, 0, 2))  # [seq_len, batch, vocab] -> [batch, seq_len, vocab]

    def forward(self, x, attention_mask):
        """
        Full forward pass through the TransformerVAE.
        """
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x, attention_mask)
        return logits, mu, logvar
