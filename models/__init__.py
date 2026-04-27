"""
Neural network models for trajectory prediction.
"""

from .lstm import LSTMTrajectoryPredictor
from .transformer import TransformerTrajectoryPredictor, PositionalEncoding
from .gnn import GNNTrajectoryPredictor
from .graph_idyom import GraphIDyOMEnhanced, CustomNodeEmbedding, create_model_with_strategy

# Backwards compatibility alias
GraphIDyOMPredictor = GraphIDyOMEnhanced

__all__ = [
    'LSTMTrajectoryPredictor',
    'TransformerTrajectoryPredictor', 
    'PositionalEncoding',
    'GNNTrajectoryPredictor',
    'GraphIDyOMEnhanced',
    'GraphIDyOMPredictor',
    'CustomNodeEmbedding',
    'create_model_with_strategy',
]
