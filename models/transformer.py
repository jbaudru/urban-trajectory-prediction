"""
Transformer model for trajectory prediction.
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTrajectoryPredictor(nn.Module):
    """Transformer model for trajectory prediction"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerTrajectoryPredictor, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        embedded = self.embedding(x) * np.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        
        # Transformer expects (seq_len, batch_size, d_model)
        embedded = embedded.transpose(0, 1)
        
        # Create attention mask for padding
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        transformer_out = self.transformer_encoder(embedded, mask)
        
        # Use the last output for prediction
        output = self.fc(self.dropout(transformer_out[-1]))
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
