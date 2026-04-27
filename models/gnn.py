"""
Graph Neural Network (GNN) model for trajectory prediction.
Uses Graph Attention Networks (GAT) for better node relationship modeling.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GNNTrajectoryPredictor(nn.Module):
    """Graph Neural Network for trajectory prediction"""
    
    def __init__(self, vocab_size, node_features_dim=128, hidden_dim=256, num_layers=4, dropout=0.2):
        super(GNNTrajectoryPredictor, self).__init__()
        
        self.node_embedding = nn.Embedding(vocab_size, node_features_dim)
        
        # Use GAT layers for better attention mechanism
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(node_features_dim, hidden_dim, heads=4, concat=False))
        
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # Bidirectional LSTM with more layers
        self.sequence_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Validate input indices
        if x.min() < 0 or x.max() >= self.node_embedding.num_embeddings:
            raise ValueError(f"Input indices out of range: [{x.min().item()}, {x.max().item()}], "
                           f"expected [0, {self.node_embedding.num_embeddings-1}]")
        
        # Create node features for the entire vocabulary
        vocab_size = self.node_embedding.num_embeddings
        all_node_indices = torch.arange(vocab_size, device=x.device, dtype=torch.long)
        all_node_features = self.node_embedding(all_node_indices)  # (vocab_size, node_features_dim)
        
        # Apply GNN layers to all nodes with residual connections
        current_features = all_node_features
        for i, layer in enumerate(self.gnn_layers):
            new_features = torch.relu(layer(current_features, edge_index))
            new_features = self.layer_norm(new_features)
            new_features = self.dropout(new_features)
            # Add residual connection for deeper layers
            if i > 0 and new_features.shape == current_features.shape:
                current_features = current_features + new_features
            else:
                current_features = new_features
        
        # Extract features for the nodes in our sequences
        x_flat = x.view(-1)  # (batch_size * seq_len,)
        sequence_features = current_features[x_flat]  # (batch_size * seq_len, hidden_dim)
        sequence_features = sequence_features.view(batch_size, seq_len, -1)
        
        # Apply sequence encoder
        lstm_out, (hidden, cell) = self.sequence_encoder(sequence_features)
        
        # Use last output for prediction
        output = self.classifier(self.dropout(lstm_out[:, -1, :]))
        return output
