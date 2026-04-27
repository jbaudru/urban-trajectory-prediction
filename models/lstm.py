"""
LSTM model for trajectory prediction.
"""

import torch
import torch.nn as nn


class LSTMTrajectoryPredictor(nn.Module):
    """LSTM model for trajectory prediction"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, dropout=0.3):
        super(LSTMTrajectoryPredictor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last output for prediction
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output
