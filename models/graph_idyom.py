"""
Enhanced GraphIDyOM with Dual LTM/STM Architecture.

This module implements a dual-model cognitive architecture inspired by IDyOM 
(Information Dynamics of Music), applied to road network trajectory prediction.

Key Architecture Components:
- Long-Term Model (LTM): Learns statistical style knowledge from the training corpus
  (road network structure/transitions). Static, global, general knowledge.
- Short-Term Model (STM): Learns piece-specific patterns from the current trajectory.
  Dynamic, local, adapted to current sequence.
- Bidirectional sequence encoder with multi-head attention for full context processing
- Variable-order Markov chains (K orders) processed sequentially per model
- Adaptive gating to balance LTM vs STM predictions based on context
- Graph-based node embeddings (degree, PageRank, or combined features)

Inspired by IDyOM (Pearce, 2005) but adapted for directed graphs and trajectory prediction.
Follows IDyOM principles: dual models, variable-order Markov, context-dependent prediction,
multiple representations, probabilistic inference.

Key Classes:
- CustomNodeEmbedding: Graph-based node embedding from structural features
- GraphIDyOMEnhanced: Main dual-model architecture with LTM + STM branches

Helper Functions:
- create_degree_features(): Degree-based embeddings
- create_pagerank_features(): PageRank-based embeddings
- create_combined_features(): Combined structural features
- create_model_with_strategy(): Factory function for model creation
- build_graph_structure_from_sequences(): Build graph from sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable
import networkx as nx


class CustomNodeEmbedding(nn.Module):
    """
    Flexible node embedding wrapper that accepts custom feature computation.
    
    Instead of learning embeddings, computes them from graph structure using
    a custom feature function.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 feature_fn: Callable[[int], np.ndarray]):
        """
        Args:
            vocab_size: Number of unique nodes
            embedding_dim: Desired embedding dimension
            feature_fn: Function that takes node_id -> np.ndarray of features
                       Will be projected to embedding_dim
        """
        super(CustomNodeEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feature_fn = feature_fn
        
        # Cache features to avoid recomputation and CPU transfers
        self._feature_cache = {}  # {node_id: feature_array}
        self._projection = None
        self._feature_dim = None
        self._initialized = False
    
    def _initialize_projection(self, feature_dim: int, device: torch.device) -> None:
        """Initialize projection matrix on first use with correct feature dimension."""
        if not self._initialized:
            self._feature_dim = feature_dim
            self._projection = nn.Linear(self._feature_dim, self.embedding_dim).to(device)
            self._initialized = True
    
    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for given nodes with caching.
        
        Args:
            node_indices: (batch_size,) node IDs
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        batch_size = node_indices.shape[0]
        device = node_indices.device
        node_ids_cpu = node_indices.cpu().tolist()
        
        # Compute raw features for each node (with caching)
        features = []
        for node_id in node_ids_cpu:
            if node_id not in self._feature_cache:
                self._feature_cache[node_id] = self.feature_fn(int(node_id))
            features.append(self._feature_cache[node_id])
        
        features = np.array(features, dtype=np.float32)  # (batch_size, feature_dim)
        
        # Initialize projection if needed
        self._initialize_projection(features.shape[1], device)
        
        # Convert to tensor and project (faster with direct from_numpy)
        features_tensor = torch.from_numpy(features).to(device)
        embeddings = self._projection(features_tensor)
        
        return embeddings


class GraphIDyOMEnhanced(nn.Module):
    """
    Enhanced GraphIDyOM with explicit LTM/STM structure inspired by IDyOM.
    
    Key improvements:
    - DUAL-MODEL ARCHITECTURE: Separate Long-Term (LTM) and Short-Term (STM) models
    - LTM: Learns style knowledge from training corpus (graph-based transitions)
    - STM: Learns piece-specific patterns from current sequence dynamically
    - Encodes FULL SEQUENCE of L previous nodes
    - Bidirectional LSTM for context encoding
    - Multi-head attention over sequence positions
    - Variable-order Markov chains (orders 1 to K) per model
    - Adaptive weighting of orders and models
    
    Usage example:
        model = GraphIDyOMEnhanced(
            vocab_size=vocab_size,
            embedding_dim=128,
            max_order=7
        )
        logits = model(sequences, graph_structure=graph_structure)
    """
    
    STRATEGIES = {
        'degree': 'Use in-degree and out-degree',
        'log_degree': 'Use log-normalized degrees',
        'pagerank': 'Use PageRank scores',
        'betweenness': 'Use betweenness centrality',
        'clustering': 'Use clustering coefficient',
        'combined': 'Use combined structural features',
        'custom': 'Use custom feature function',
    }
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_order: int = 10,
                 embedding_strategy: str = 'degree',
                 custom_feature_fn: Optional[Callable] = None,
                 use_neural_weighting: bool = True,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Node embedding dimension (default 128)
            max_order: Maximum order for Markov chains
            embedding_strategy: Which embedding to use
            custom_feature_fn: Function for custom embeddings
            use_neural_weighting: Whether to learn order weights
            dropout: Dropout rate
        """
        super(GraphIDyOMEnhanced, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_order = max_order
        self.embedding_strategy = embedding_strategy
        self.use_neural_weighting = use_neural_weighting
        
        # ====== OPTIMIZATION: Pre-compute decay weights for all orders ======
        # Avoid redundant exponential calculations during forward pass
        self._decay_rate = 0.5
        self._precomputed_decay_weights = self._compute_all_decay_weights(max_order)
        
        # ====== 1. NODE EMBEDDING ======
        if embedding_strategy == 'custom' and custom_feature_fn is not None:
            self.embedding = CustomNodeEmbedding(
                vocab_size, embedding_dim, custom_feature_fn
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ====== 2. SEQUENCE ENCODER (Process full historical context) ======
        self.sequence_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
            bidirectional=True
        )
        sequence_encoder_output_dim = 256 * 2  # bidirectional = 512
        
        # ====== 3. ATTENTION OVER SEQUENCE POSITIONS ======
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=sequence_encoder_output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ====== 4. LTM REPRESENTATION (Graph-based, fixed style knowledge) ======
        # Projects Markov distributions to representation space
        # Direct processing without LSTM: average K orders then project
        self.ltm_projector = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ====== 5. STM REPRESENTATION (Sequence-based, dynamic pattern learning) ======
        # Projects Markov distributions to representation space
        # Direct processing without LSTM: average K orders then project
        self.stm_projector = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ====== 6. MODEL GATING (Adaptive weighting of LTM vs STM) ======
        # Learn to balance long-term vs short-term knowledge
        self.model_gate = nn.Sequential(
            nn.Linear(sequence_encoder_output_dim + 512, 256),  # seq_repr + ltm+stm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # 2 outputs: [alpha_ltm, alpha_stm]
            nn.Softmax(dim=1)
        )
        
        # ====== 7. CONTEXT FUSION ======
        # Combine sequence context + weighted LTM/STM representations
        fusion_input_dim = sequence_encoder_output_dim + 256  # seq + (weighted ltm+stm)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, vocab_size)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _compute_all_decay_weights(self, max_order: int) -> Dict[int, np.ndarray]:
        """
        OPTIMIZATION: Pre-compute decay weights for all orders to avoid redundant
        exponential calculations during inference. Decay weights are order-dependent
        but independent of sequence content, so compute once at init.
        
        Args:
            max_order: Maximum Markov order
            
        Returns:
            Dictionary mapping order -> normalized decay weights array
        """
        decay_weights_dict = {}
        decay_rate = self._decay_rate
        
        for order in range(1, max_order + 1):
            # Exponential decay: recent positions get higher weight
            order_weights = np.exp(-decay_rate * np.arange(order, dtype=np.float32))
            order_weights = order_weights / order_weights.sum()
            order_weights = order_weights[::-1].copy()  # Reverse to have most recent first (copy to avoid negative strides)
            decay_weights_dict[order] = order_weights
        
        return decay_weights_dict
    
    def forward(self, sequences: torch.Tensor,
                graph_structure: Optional[Dict] = None) -> torch.Tensor:
        """
        Predict next node using dual LTM/STM architecture.
        
        Args:
            sequences: (batch_size, seq_len) node sequences
            graph_structure: Dict with graph-based transitions (for LTM)
            
        Returns:
            logits: (batch_size, vocab_size)
        """
        batch_size = sequences.shape[0]
        device = sequences.device
        seq_len = sequences.shape[1]
        
        # ====== STEP 1: EMBED FULL SEQUENCE ======
        embedded_seq = self.embedding(sequences)  # (batch, seq_len, embedding_dim)
        
        # ====== STEP 2: ENCODE SEQUENCE WITH BIDIRECTIONAL LSTM ======
        lstm_output, (h_n, c_n) = self.sequence_encoder(embedded_seq)
        
        # ====== STEP 3: ATTENTION OVER SEQUENCE ======
        # OPTIMIZATION: Don't store unused attention weights
        attn_output, _ = self.sequence_attention(
            lstm_output, lstm_output, lstm_output
        )
        seq_representation = attn_output.mean(dim=1)  # (batch, 512)
        
        # ====== STEP 4: LTM BRANCH (Graph-based, style knowledge) ======
        # Compute Markov orders from training graph structure (fixed, global knowledge)
        ltm_markov_distributions = self._compute_markov_from_graph(sequences, graph_structure)
        
        # OPTIMIZATION: Compute mean directly without intermediate stacking
        ltm_avg_dist = torch.stack(ltm_markov_distributions, dim=0).mean(dim=0)
        
        # Project to representation space
        ltm_representation = self.ltm_projector(ltm_avg_dist)  # (batch, 256)
        
        # ====== STEP 5: STM BRANCH (Sequence-based, piece-specific knowledge) ======
        # Compute Markov orders from current sequence (dynamic, local knowledge)
        stm_markov_distributions = self._compute_markov_from_sequences(sequences)
        
        # OPTIMIZATION: Compute mean directly without intermediate stacking
        stm_avg_dist = torch.stack(stm_markov_distributions, dim=0).mean(dim=0)
        
        # Project to representation space
        stm_representation = self.stm_projector(stm_avg_dist)  # (batch, 256)
        
        # ====== STEP 6: ADAPTIVE MODEL GATING (Balance LTM vs STM) ======
        # Learn to weight which model is more important for this context
        gate_input = torch.cat([seq_representation, ltm_representation, stm_representation], dim=1)
        model_weights = self.model_gate(gate_input)  # (batch, 2) -> softmax
        # OPTIMIZATION: Keep weights as (batch, 1) for in-place multiplication
        alpha_ltm_w = model_weights[:, 0:1]  # (batch, 1) - more efficient than unsqueeze
        alpha_stm_w = model_weights[:, 1:2]  # (batch, 1)
        
        # Weighted combination: alpha * LTM + (1-alpha) * STM
        combined_representation = alpha_ltm_w * ltm_representation + alpha_stm_w * stm_representation
        
        # ====== STEP 7: FUSE SEQUENCE AND MODEL INFORMATION ======
        fused = torch.cat([seq_representation, combined_representation], dim=1)
        
        # Process through fusion network
        logits = self.fusion(fused)  # (batch, vocab_size)
        
        return logits
    
    def forward_with_explanations(self, sequences: torch.Tensor,
                                  graph_structure: Optional[Dict] = None) -> Dict:
        """
        Forward pass that returns intermediate LTM/STM probabilities for explainability.
        
        Returns:
            Dict with keys:
                - 'logits': (batch_size, vocab_size) - final predictions
                - 'ltm_probs': (batch_size, vocab_size) - softmax of LTM probabilities
                - 'stm_probs': (batch_size, vocab_size) - softmax of STM probabilities
                - 'alpha_ltm': (batch_size,) - gate weight for LTM (0 to 1)
                - 'alpha_stm': (batch_size,) - gate weight for STM (0 to 1)
                - 'ltm_representation': (batch_size, 256) - LTM representation
                - 'stm_representation': (batch_size, 256) - STM representation
        """
        batch_size = sequences.shape[0]
        device = sequences.device
        seq_len = sequences.shape[1]
        
        # ====== STEP 1: EMBED FULL SEQUENCE ======
        embedded_seq = self.embedding(sequences)
        
        # ====== STEP 2: ENCODE SEQUENCE WITH BIDIRECTIONAL LSTM ======
        lstm_output, (h_n, c_n) = self.sequence_encoder(embedded_seq)
        
        # ====== STEP 3: ATTENTION OVER SEQUENCE ======
        attn_output, _ = self.sequence_attention(lstm_output, lstm_output, lstm_output)
        seq_representation = attn_output.mean(dim=1)
        
        # ====== STEP 4: LTM BRANCH (with softmax for probabilities) ======
        ltm_markov_distributions = self._compute_markov_from_graph(sequences, graph_structure)
        ltm_avg_dist = torch.stack(ltm_markov_distributions, dim=0).mean(dim=0)
        ltm_probs = F.softmax(ltm_avg_dist, dim=1)  # Convert to probabilities
        ltm_representation = self.ltm_projector(ltm_avg_dist)
        
        # ====== STEP 5: STM BRANCH (with softmax for probabilities) ======
        stm_markov_distributions = self._compute_markov_from_sequences(sequences)
        stm_avg_dist = torch.stack(stm_markov_distributions, dim=0).mean(dim=0)
        stm_probs = F.softmax(stm_avg_dist, dim=1)  # Convert to probabilities
        stm_representation = self.stm_projector(stm_avg_dist)
        
        # ====== STEP 6: ADAPTIVE MODEL GATING ======
        gate_input = torch.cat([seq_representation, ltm_representation, stm_representation], dim=1)
        model_weights = self.model_gate(gate_input)  # (batch, 2) -> softmax
        alpha_ltm = model_weights[:, 0]  # (batch,)
        alpha_stm = model_weights[:, 1]  # (batch,)
        
        # ====== STEP 7: FUSE AND PREDICT ======
        combined_representation = alpha_ltm.unsqueeze(1) * ltm_representation + \
                                  alpha_stm.unsqueeze(1) * stm_representation
        fused = torch.cat([seq_representation, combined_representation], dim=1)
        logits = self.fusion(fused)
        
        return {
            'logits': logits,
            'ltm_probs': ltm_probs,
            'stm_probs': stm_probs,
            'alpha_ltm': alpha_ltm,
            'alpha_stm': alpha_stm,
            'ltm_representation': ltm_representation,
            'stm_representation': stm_representation,
        }
    
    def _compute_markov_from_graph(self, sequences: torch.Tensor, 
                                   graph_structure: Dict) -> List[torch.Tensor]:
        """
        LTM: Compute high-order Markov distributions from GRAPH STRUCTURE.
        
        This represents LONG-TERM style knowledge learned from the training corpus.
        Returns the same distributions for all samples in the batch (global knowledge).
        Uses pre-computed decay weights for significant speedup.
        
        For each order k (1 to max_order):
        - Uses the last k nodes in the sequence
        - Weights more recent nodes more heavily (exponential decay)
        - Averages their transition probabilities from graph
        - Higher orders = more historical context
        
        Args:
            sequences: (batch_size, seq_len) node sequences
            graph_structure: Dict with edge_weights from training data
            
        Returns:
            List of K Markov distributions (LTM knowledge)
        """
        if graph_structure is None:
            # Fallback to uniform if no graph provided
            return self._compute_markov_from_sequences(sequences)
            
        edge_weights = graph_structure['edge_weights']
        batch_size = sequences.shape[0]
        vocab_size = self.vocab_size
        device = sequences.device
        seq_len = sequences.shape[1]
        
        # Normalize edge weights to create transition probabilities
        transition_matrix = edge_weights / (edge_weights.sum(dim=1, keepdim=True) + 1e-8)
        transition_matrix = transition_matrix.to(device)
        
        markov_distributions = []
        
        # Compute Markov distributions for orders 1 to max_order
        for order in range(1, self.max_order + 1):
            if order <= seq_len:
                nodes_for_order = sequences[:, -order:]  # (batch_size, order)
                
                # OPTIMIZATION: Use pre-computed decay weights instead of computing each time
                order_weights_np = self._precomputed_decay_weights[order]
                order_weights = torch.from_numpy(order_weights_np).to(device)
                
                # OPTIMIZATION: Vectorized computation instead of loop over positions
                # Shape: (batch_size, order) x (order,) -> (batch_size, order)
                weighted_nodes = nodes_for_order  # (batch_size, order)
                
                # Gather transition probs for each node and weight them
                # transition_matrix[nodes_for_order] -> (batch, order, vocab)
                transitions = transition_matrix[nodes_for_order]  # (batch, order, vocab)
                # Weight each transition and sum: (batch, vocab)
                order_dist = (transitions * order_weights.unsqueeze(0).unsqueeze(2)).sum(dim=1)
                
                markov_distributions.append(order_dist)
            else:
                last_nodes = sequences[:, -1]
                markov_distributions.append(transition_matrix[last_nodes, :])
        
        return markov_distributions
    
    def _compute_markov_from_sequences(self, sequences: torch.Tensor) -> List[torch.Tensor]:
        """
        STM: Compute high-order Markov distributions DYNAMICALLY from CURRENT SEQUENCE.
        
        This represents SHORT-TERM model learning from piece-specific patterns.
        Learns variable transition probabilities unique to this trajectory.
        Uses pre-computed decay weights for efficiency and FULLY VECTORIZED.
        
        For each order k (1 to max_order):
        - Looks back k steps in the current sequence
        - Builds transition probabilities from within the sequence
        - Weights more recent patterns more heavily
        - Adapts to the specific structure of this path
        
        Args:
            sequences: (batch_size, seq_len) node sequences
            
        Returns:
            List of K Markov distributions (STM knowledge) - piece-specific
        """
        batch_size = sequences.shape[0]
        device = sequences.device
        seq_len = sequences.shape[1]
        vocab_size = self.vocab_size
        
        markov_distributions = []
        
        # For each order, compute transitions from within the current sequence
        for order in range(1, self.max_order + 1):
            if order <= seq_len:
                # Get the last 'order' nodes
                nodes_for_order = sequences[:, -order:]  # (batch_size, order)
                
                # OPTIMIZATION: Use pre-computed decay weights
                order_weights_np = self._precomputed_decay_weights[order]
                order_weights = torch.from_numpy(order_weights_np).to(device)  # (order,)
                
                # OPTIMIZATION: Vectorized weighted counting using broadcasting
                # Reshape weights to broadcast: (order,) -> (1, order)
                weights_broadcasted = order_weights.unsqueeze(0)  # (1, order)
                
                # Weighted counts: (batch, order) * (1, order) -> (batch, order)
                weighted_nodes = nodes_for_order * 0 + order_weights.unsqueeze(0)  # Apply weights
                
                # Use scatter_add for each batch independently (vectorized)
                node_counts = torch.zeros(batch_size, vocab_size, device=device, dtype=torch.float32)
                
                # Reshape for scatter_add: flatten batch dimension
                nodes_flat = nodes_for_order.reshape(-1)  # (batch * order,)
                weights_flat = order_weights.unsqueeze(0).expand(batch_size, order).reshape(-1)  # (batch * order,)
                batch_indices_flat = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, order).reshape(-1)
                
                # Validate indices
                valid_mask = (nodes_flat >= 0) & (nodes_flat < vocab_size)
                nodes_flat = nodes_flat[valid_mask]
                weights_flat = weights_flat[valid_mask]
                batch_indices_flat = batch_indices_flat[valid_mask]
                
                # Scatter add: accumulate weights for each node in each batch
                node_counts[batch_indices_flat, nodes_flat] += weights_flat
                
                # Normalize to get probability distribution
                node_sums = node_counts.sum(dim=1, keepdim=True) + 1e-8
                order_dist = node_counts / node_sums  # (batch, vocab)
                
            else:
                # If we don't have enough history, use uniform
                order_dist = torch.ones(batch_size, vocab_size, device=device) / vocab_size
            
            markov_distributions.append(order_dist)
        
        return markov_distributions
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """Get available strategies and their descriptions."""
        return cls.STRATEGIES.copy()


# ============================================================================
# HELPER FUNCTIONS: Strategy Implementations
# ============================================================================

def create_degree_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for degree-based embedding.
    
    Features: normalized in-degree and out-degree
    """
    in_degrees = np.zeros(vocab_size)
    out_degrees = np.zeros(vocab_size)
    
    for node in graph.nodes():
        if node < vocab_size:
            in_degrees[node] = graph.in_degree(node)
            out_degrees[node] = graph.out_degree(node)
    
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    out_degrees = out_degrees / (out_degrees.max() + 1e-6)
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([in_degrees[node_id], out_degrees[node_id]])
    
    return fn


def create_pagerank_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for PageRank-based embedding.
    
    Features: PageRank scores and in-degree
    """
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except:
        pagerank = {i: 1.0 / vocab_size for i in range(vocab_size)}
    
    pagerank_arr = np.array([pagerank.get(i, 1.0/vocab_size) for i in range(vocab_size)])
    pagerank_arr = pagerank_arr / (pagerank_arr.max() + 1e-6)
    
    in_degrees = np.array([graph.in_degree(i) for i in range(vocab_size)])
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([pagerank_arr[node_id], in_degrees[node_id]])
    
    return fn


def create_combined_features(graph: nx.DiGraph, vocab_size: int) -> Callable:
    """
    Create feature function for combined features.
    
    Features: in-degree, out-degree, PageRank, and clustering coefficient
    """
    # Degree
    in_degrees = np.array([graph.in_degree(i) for i in range(vocab_size)])
    out_degrees = np.array([graph.out_degree(i) for i in range(vocab_size)])
    in_degrees = in_degrees / (in_degrees.max() + 1e-6)
    out_degrees = out_degrees / (out_degrees.max() + 1e-6)
    
    # PageRank
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except:
        pagerank = {i: 1.0 / vocab_size for i in range(vocab_size)}
    pagerank_arr = np.array([pagerank.get(i, 1.0/vocab_size) for i in range(vocab_size)])
    pagerank_arr = pagerank_arr / (pagerank_arr.max() + 1e-6)
    
    # Clustering
    undirected = graph.to_undirected()
    clustering = np.array([nx.clustering(undirected, i) if i in undirected else 0.0 
                          for i in range(vocab_size)])
    
    def fn(node_id: int) -> np.ndarray:
        return np.array([in_degrees[node_id], out_degrees[node_id], 
                        pagerank_arr[node_id], clustering[node_id]])
    
    return fn


# Mapping of strategy names to feature functions
FEATURE_CREATORS = {
    'degree': create_degree_features,
    'pagerank': create_pagerank_features,
    'combined': create_combined_features,
}


def create_model_with_strategy(
    graph: nx.DiGraph,
    vocab_size: int,
    embedding_strategy: str = 'degree',
    embedding_dim: int = 128,
    max_order: int = 7,
    use_neural_weighting: bool = True
) -> GraphIDyOMEnhanced:
    """
    Create GraphIDyOM model with specified embedding strategy.
    
    This is a convenience function that simplifies model creation with different
    embedding strategies, automatically setting up the feature function based on
    the graph structure.
    
    Args:
        graph: NetworkX DiGraph
        vocab_size: Vocabulary size (number of unique nodes)
        embedding_strategy: Which strategy to use ('degree', 'pagerank', 'combined')
        embedding_dim: Embedding dimension (default 128 for richer representation)
        max_order: Max Markov order (default 7 for higher-order chains)
        use_neural_weighting: Whether to learn order weights
        
    Returns:
        Initialized GraphIDyOMEnhanced model ready for training
        
    Example:
        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        >>> model = create_model_with_strategy(graph, vocab_size=3, embedding_strategy='degree')
    """
    # Get feature function
    if embedding_strategy in FEATURE_CREATORS:
        feature_fn = FEATURE_CREATORS[embedding_strategy](graph, vocab_size)
    else:
        # Default to degree
        feature_fn = FEATURE_CREATORS['degree'](graph, vocab_size)
    
    # Create model
    model = GraphIDyOMEnhanced(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_order=max_order,
        embedding_strategy='custom',
        custom_feature_fn=feature_fn,
        use_neural_weighting=use_neural_weighting
    )
    
    return model


# ============================================================================
# HELPER: Build graph structure from sequences
# ============================================================================

def build_graph_structure_from_sequences(sequences: torch.Tensor, vocab_size: int) -> Dict:
    """
    Build graph structure (adjacency, edge weights, degrees) from node sequences.
    FULLY VECTORIZED - no Python loops for efficiency on GPU.
    
    Args:
        sequences: (N, seq_len) tensor of node indices
        vocab_size: Number of unique nodes
        
    Returns:
        Dictionary with keys:
            - 'adjacency': (vocab_size, vocab_size) adjacency matrix
            - 'edge_weights': (vocab_size, vocab_size) weighted adjacency
            - 'in_degrees': (vocab_size,) in-degree for each node
            - 'out_degrees': (vocab_size,) out-degree for each node
    """
    device = sequences.device
    
    # OPTIMIZATION: Fully vectorized edge counting without Python loops
    # Extract source and target nodes using slicing
    batch_size, seq_len = sequences.shape
    
    # Get all from_nodes and to_nodes from consecutive pairs
    from_nodes = sequences[:, :-1].reshape(-1)  # (batch_size * (seq_len - 1),)
    to_nodes = sequences[:, 1:].reshape(-1)      # (batch_size * (seq_len - 1),)
    
    # Filter out invalid node indices
    valid_mask = (from_nodes >= 0) & (from_nodes < vocab_size) & (to_nodes >= 0) & (to_nodes < vocab_size)
    from_nodes = from_nodes[valid_mask]
    to_nodes = to_nodes[valid_mask]
    
    # Count edges using scatter_add (MUCH faster than looping)
    edge_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
    edge_indices = from_nodes * vocab_size + to_nodes
    edge_counts.view(-1).scatter_add_(0, edge_indices, torch.ones_like(from_nodes, dtype=torch.float32))
    
    # Convert counts to probabilities (edge weights)
    row_sums = edge_counts.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    edge_weights = edge_counts / row_sums
    
    # Binary adjacency (any edge exists if count > 0)
    adjacency = (edge_counts > 0).float()
    
    # Add self-loops
    adjacency.fill_diagonal_(1.0)
    edge_weights.fill_diagonal_(0.5)  # Give self-loops higher weight
    
    # Calculate degrees from adjacency
    in_degrees = adjacency.sum(dim=0)
    out_degrees = adjacency.sum(dim=1)
    
    return {
        'adjacency': adjacency,
        'edge_weights': edge_weights,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
    }
