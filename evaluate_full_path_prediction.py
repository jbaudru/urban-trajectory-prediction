"""
Evaluate full-path prediction metrics using autoregressive inference.

This script expects a `models.py` module in the same directory with:
- LSTMTrajectoryPredictor
- TransformerTrajectoryPredictor  
- GNNTrajectoryPredictor
- GraphIDyOMPredictor

Each model should support:
- __init__(vocab_size, sequence_length=10, ...)
- forward(input_ids) -> logits of shape (batch, seq_len, vocab_size) or (batch, vocab_size)
- to(device)
- eval()

For each test route: seed with first L nodes, iteratively predict next node
using greedy decoding (argmax), shift context window, repeat until path 
length matches ground truth.
"""

from __future__ import annotations

import json
import csv
import argparse
from pathlib import Path
from typing import Any, Optional
import random
import traceback

import numpy as np
import torch
from difflib import SequenceMatcher
from sklearn.metrics import f1_score, average_precision_score


def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein distance between two sequences."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    
    if len(seq2) == 0:
        return len(seq1)
    
    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_edit_distance(seq1, seq2):
    """Compute normalized Levenshtein distance."""
    lev_dist = levenshtein_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return lev_dist / max_len if max_len > 0 else 0.0


class AutoregressiveEvaluator:
    """Evaluator for full-path autoregressive prediction."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.edge_index = None  # For GNN models
        self.model_type = None  # Track model type for special handling
        self.graph_struct = None  # For GraphIDyOM: adjacency, edge_weights, degrees
        self.vocab_size = None  # Test set vocabulary size
        self.model_vocab_size = None  # Model training vocabulary size (CRITICAL: for clamping predictions)
        
    @torch.no_grad()
    def predict_next_node(self, model: torch.nn.Module, context: torch.Tensor) -> int:
        """
        Predict next node using greedy decoding (argmax).
        
        Args:
            model: The trained model in eval mode
            context: Tensor of shape (1, L) containing node indices (already embedded by model)
            
        Returns:
            Predicted node index
        """
        context = context.to(self.device)
        
        # CRITICAL: Clamp input context to model's vocab range BEFORE embedding lookup
        # This prevents CUDA indexing errors when context contains indices >= model_vocab_size
        if self.model_vocab_size is not None and context.max().item() >= self.model_vocab_size:
            context = torch.clamp(context, max=self.model_vocab_size - 1)
        
        # Forward pass with model-specific arguments
        if self.model_type == "gnn" and self.edge_index is not None:
            # GNN needs edge connectivity information
            output = model(context, self.edge_index.to(self.device))
        elif self.model_type == "graphidyom" and self.graph_struct is not None:
            # GraphIDyOM needs graph structure (adjacency, edge_weights, degrees)
            # Move graph_struct tensors to device
            graph_struct_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.graph_struct.items()
            }
            output = model(context, graph_struct_device)
        else:
            # LSTM, Transformer: just need token indices (embedding in model)
            output = model(context)
        
        # Get logits
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Handle different output shapes
        if logits.dim() == 3:
            # Shape: (batch, seq_len, vocab_size)
            last_step_logits = logits[0, -1, :]
        elif logits.dim() == 2:
            # Shape: (batch, vocab_size)
            last_step_logits = logits[0, :]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        # Greedy decoding: argmax
        predicted_node = torch.argmax(last_step_logits).item()
        
        # CRITICAL: Clamp to model's training vocab size, NOT test set vocab size
        # Models trained with vocab_size=3508 can only embed indices [0, 3508)
        # Even if test set has vocab_size=3511, we must clamp to 3507 to avoid embedding lookup errors
        if self.model_vocab_size is not None and predicted_node >= self.model_vocab_size:
            predicted_node = self.model_vocab_size - 1
        
        return predicted_node

    @torch.no_grad()
    def predict_next_node_with_probs(
        self, model: torch.nn.Module, context: torch.Tensor
    ) -> tuple[int, np.ndarray]:
        """
        Predict next node and return full softmax probability distribution.

        Returns:
            Tuple of (predicted_node_index, prob_array of shape (vocab_size,))
        """
        context = context.to(self.device)

        if self.model_vocab_size is not None and context.max().item() >= self.model_vocab_size:
            context = torch.clamp(context, max=self.model_vocab_size - 1)

        if self.model_type == "gnn" and self.edge_index is not None:
            output = model(context, self.edge_index.to(self.device))
        elif self.model_type == "graphidyom" and self.graph_struct is not None:
            graph_struct_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.graph_struct.items()
            }
            output = model(context, graph_struct_device)
        else:
            output = model(context)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            last_step_logits = logits[0, -1, :]
        elif logits.dim() == 2:
            last_step_logits = logits[0, :]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        probs = torch.softmax(last_step_logits, dim=-1).cpu().numpy()
        predicted_node = int(np.argmax(probs))

        if self.model_vocab_size is not None and predicted_node >= self.model_vocab_size:
            predicted_node = self.model_vocab_size - 1

        return predicted_node, probs

    def autoregressive_inference(
        self,
        model: torch.nn.Module,
        seed_path: list[int],
        target_length: int,
        max_new_tokens: int = 500,
        return_probs: bool = False,
    ) -> list[int] | tuple[list[int], list[np.ndarray]]:
        """
        Perform autoregressive inference to generate full path.
        
        Args:
            model: Trained model in eval mode
            seed_path: Initial nodes (first L nodes)
            target_length: Desired path length
            max_new_tokens: Maximum tokens to generate
            return_probs: If True, also return per-step softmax probability arrays
            
        Returns:
            Full predicted path, or (path, list_of_prob_arrays) when return_probs=True
        """
        context = seed_path.copy()
        sequence_length = len(seed_path)
        all_probs: list[np.ndarray] = []
        
        # Generate until we reach target length
        for step in range(min(target_length - sequence_length, max_new_tokens)):
            # Create tensor from context (use last `sequence_length` tokens)
            context_tensor = torch.tensor([context[-sequence_length:]], dtype=torch.long)
            
            # Predict next node
            try:
                if return_probs:
                    pred_node, probs = self.predict_next_node_with_probs(model, context_tensor)
                    all_probs.append(probs)
                else:
                    pred_node = self.predict_next_node(model, context_tensor)
                context.append(pred_node)
            except Exception as e:
                print(f"Error during prediction: {e}")
                break
        
        if return_probs:
            return context, all_probs
        return context
    
    def evaluate_on_paths(
        self,
        model: torch.nn.Module,
        paths: list[list[int]],
        sequence_lengths: list[int] = [2, 5, 10],
        num_samples: int = 100,
        seed: int = 42,
        model_vocab_size: Optional[int] = None,
    ) -> dict[int, dict[str, float]]:
        """
        Evaluate model on a set of paths for different sequence lengths.
        
        Args:
            model: Trained model in eval mode
            paths: List of ground-truth paths (as node sequences)
            sequence_lengths: List of L values to evaluate
            num_samples: Number of paths to sample for evaluation
            seed: Random seed for sampling
            model_vocab_size: Vocabulary size the model was trained on (for filtering paths)
            
        Returns:
            Dictionary mapping L -> {mean_node_accuracy, std_node_accuracy, 
                                      mean_normalized_edit_distance, std_normalized_edit_distance}
        """
        random.seed(seed)
        results = {}
        
        # Store model vocab size for use during prediction (CRITICAL!)
        self.model_vocab_size = model_vocab_size
        
        # CRITICAL: Filter edge_index to only use nodes within model's training vocabulary
        # This prevents CUDA indexing errors when edge_index references nodes >= model_vocab_size
        if model_vocab_size is not None and model_vocab_size < self.vocab_size:
            if self.edge_index is not None and self.edge_index.numel() > 0:
                # Keep only edges where both source and target are < model_vocab_size
                valid_edges = (self.edge_index[0] < model_vocab_size) & (self.edge_index[1] < model_vocab_size)
                self.edge_index = self.edge_index[:, valid_edges]
                print(f"  Filtered edge_index: kept {valid_edges.sum().item()} valid edges (vocab {model_vocab_size})")
        
        # Filter paths to only include valid indices
        valid_paths = paths
        if model_vocab_size is not None and model_vocab_size < self.vocab_size:
            # Some nodes in test set may not exist in training vocab
            # Only use paths where all nodes are in model's training vocabulary
            valid_paths = [
                path for path in paths 
                if all(node < model_vocab_size for node in path)
            ]
            skipped = len(paths) - len(valid_paths)
            if skipped > 0:
                print(f"  Filtered out {skipped} paths with nodes outside model vocab (>{model_vocab_size-1})")
        
        # Sample paths
        sampled_paths = random.sample(valid_paths, min(num_samples, len(valid_paths)))
        print(f"  Sampling {len(sampled_paths)} paths for evaluation")
        
        for L in sequence_lengths:
            accuracies = []
            edit_distances = []
            all_true_nodes: list[int] = []
            all_pred_nodes: list[int] = []
            all_step_aps: list[float] = []
            valid_count = 0
            
            for path_idx, ground_truth_path in enumerate(sampled_paths):
                # Skip paths shorter than L + 1 (need at least seed + 1 target)
                if len(ground_truth_path) <= L:
                    continue
                
                valid_count += 1
                
                # Seed with first L nodes
                seed_path = ground_truth_path[:L]
                target_length = len(ground_truth_path)
                
                try:
                    # Autoregressive inference (with probabilities for AUPRC/F1)
                    predicted_path, step_probs = self.autoregressive_inference(
                        model=model,
                        seed_path=seed_path,
                        target_length=target_length,
                        return_probs=True,
                    )
                    
                    # Truncate to target length
                    predicted_path = predicted_path[:target_length]
                    
                    # Compute node accuracy
                    correct_nodes = sum(
                        1 for i in range(len(predicted_path))
                        if predicted_path[i] == ground_truth_path[i]
                    )
                    node_acc = correct_nodes / len(ground_truth_path) if len(ground_truth_path) > 0 else 0.0
                    accuracies.append(node_acc)
                    
                    # Compute normalized edit distance
                    ned = normalized_edit_distance(predicted_path, ground_truth_path)
                    edit_distances.append(ned)

                    # Collect per-step predictions and probs for F1 / AUPRC
                    # Only count generated steps (skip the seed)
                    for step_i, (pred_node, prob) in enumerate(zip(predicted_path[L:], step_probs)):
                        true_node = ground_truth_path[L + step_i]
                        all_true_nodes.append(true_node)
                        all_pred_nodes.append(pred_node)

                        # Per-step average precision (binary: true class vs rest)
                        prob_len = len(prob)
                        binary_true = np.zeros(prob_len, dtype=np.int32)
                        if true_node < prob_len:
                            binary_true[true_node] = 1
                        # Only compute AP if positive class is present
                        if binary_true.sum() > 0:
                            try:
                                ap = average_precision_score(binary_true, prob)
                                all_step_aps.append(float(ap))
                            except ValueError:
                                pass
                    
                except Exception as e:
                    print(f"    Error evaluating path {path_idx}: {e}")
                    continue
            
            # Aggregate metrics
            if accuracies:
                # F1 score (weighted across all generated steps)
                if all_true_nodes:
                    f1 = float(f1_score(
                        all_true_nodes, all_pred_nodes,
                        average='weighted', zero_division=0
                    ))
                else:
                    f1 = 0.0

                # AUPRC: mean per-step average precision
                auprc = float(np.mean(all_step_aps)) if all_step_aps else 0.0

                results[L] = {
                    "mean_node_accuracy": float(np.mean(accuracies)),
                    "std_node_accuracy": float(np.std(accuracies)),
                    "mean_normalized_edit_distance": float(np.mean(edit_distances)),
                    "std_normalized_edit_distance": float(np.std(edit_distances)),
                    "f1_score": f1,
                    "auprc": auprc,
                    "num_samples": len(accuracies),
                }
            else:
                results[L] = {
                    "mean_node_accuracy": 0.0,
                    "std_node_accuracy": 0.0,
                    "mean_normalized_edit_distance": 1.0,
                    "std_normalized_edit_distance": 0.0,
                    "f1_score": 0.0,
                    "auprc": 0.0,
                    "num_samples": 0,
                }
            
            print(f"    L={L}: {len(accuracies)} samples, "
                  f"Acc={results[L]['mean_node_accuracy']:.4f}, "
                  f"EditDist={results[L]['mean_normalized_edit_distance']:.4f}, "
                  f"F1={results[L]['f1_score']:.4f}, "
                  f"AUPRC={results[L]['auprc']:.4f}")
        
        return results
    
    def build_graph_structure_for_graphidyom(self, paths: list[list[int]], vocab_size: int) -> dict:
        """
        Build graph structure (adjacency, edge weights, degrees) for GraphIDyOM.
        Handles variable-length sequences by building from path transitions.
        
        Args:
            paths: List of node index sequences (variable length)
            vocab_size: Total number of unique nodes
            
        Returns:
            Dictionary with 'adjacency', 'edge_weights', 'in_degrees', 'out_degrees'
        """
        print(f"    Building graph structure from {len(paths)} paths with vocab_size={vocab_size}")
        
        # Build adjacency matrix from path transitions
        # This works with variable-length sequences
        adjacency = torch.zeros((vocab_size, vocab_size), device=self.device, dtype=torch.float32)
        
        for path in paths:
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                if src < vocab_size and dst < vocab_size:
                    adjacency[src, dst] += 1.0
        
        # Normalize by outgoing edges (row-stochastic)
        row_sums = adjacency.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        adjacency_normalized = adjacency / row_sums
        
        # Compute degrees
        in_degrees = adjacency.sum(dim=0)  # Column sums
        out_degrees = adjacency.sum(dim=1)  # Row sums
        
        print(f"    [OK] Graph structure: {adjacency.count_nonzero().item()} non-zero entries in adjacency")
        
        return {
            'adjacency': adjacency_normalized,
            'edge_weights': adjacency_normalized,
            'in_degrees': in_degrees,
            'out_degrees': out_degrees,
        }


def load_paths_from_csv(csv_path: Path) -> list[list[int]]:
    """Load paths from CSV file."""
    paths = []
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'q_path' in row and row['q_path'].strip():
                try:
                    path = [int(node) for node in row['q_path'].split(',')]
                    if len(path) >= 2:
                        paths.append(path)
                except (ValueError, KeyError):
                    continue
    
    return paths


def build_node_vocab(paths: list[list[int]]) -> tuple[dict[int, int], dict[int, int], int]:
    """Build vocabulary mapping from paths."""
    unique_nodes = set()
    for path in paths:
        unique_nodes.update(path)
    
    sorted_nodes = sorted(unique_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    return node_to_idx, idx_to_node, len(unique_nodes)


def load_model_checkpoint(
    checkpoint_path: Path,
    model_class: type,
    config: dict,
    device: str,
) -> Optional[tuple[torch.nn.Module, int]]:
    """Load model from checkpoint.
    
    Returns:
        Tuple of (model, checkpoint_vocab_size) or None if loading failed
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        
        # Infer vocab_size AND embedding_dim from checkpoint embedding layer
        # Different models use different embedding layer names
        checkpoint_vocab_size = None
        checkpoint_embedding_dim = None
        
        # Try common embedding layer names
        embedding_layer_names = ['embedding.weight', 'node_embedding.weight', 'token_embedding.weight']
        for layer_name in embedding_layer_names:
            if layer_name in state_dict:
                # Layer shape is (vocab_size, embedding_dim)
                checkpoint_vocab_size = state_dict[layer_name].shape[0]
                checkpoint_embedding_dim = state_dict[layer_name].shape[1]
                print(f"      Inferred vocab_size={checkpoint_vocab_size}, embedding_dim={checkpoint_embedding_dim} from '{layer_name}'")
                config['vocab_size'] = checkpoint_vocab_size
                # Add embedding_dim if model supports it
                if checkpoint_embedding_dim is not None:
                    config['embedding_dim'] = checkpoint_embedding_dim
                break
        
        if checkpoint_vocab_size is None:
            # Fallback: try to infer from classifier/fc layer
            classifier_names = ['classifier.weight', 'fc.weight']
            for layer_name in classifier_names:
                if layer_name in state_dict:
                    # classifier.weight has shape (vocab_size, hidden_dim)
                    checkpoint_vocab_size = state_dict[layer_name].shape[0]
                    print(f"      Inferred vocab_size={checkpoint_vocab_size} from '{layer_name}'")
                    config['vocab_size'] = checkpoint_vocab_size
                    break
        
        if checkpoint_vocab_size is None:
            print(f"      ⚠️  Could not infer vocab_size from checkpoint, using from config")
            checkpoint_vocab_size = config.get('vocab_size')
        
        # Instantiate model with correct vocab_size and embedding_dim
        try:
            model = model_class(**config)
        except TypeError as te:
            # If model doesn't support embedding_dim, remove it and retry
            if 'embedding_dim' in config:
                print(f"      Model doesn't accept embedding_dim, removing from config")
                config.pop('embedding_dim', None)
                model = model_class(**config)
            else:
                raise
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, checkpoint_vocab_size
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        return None


def main():
    """Main evaluation routine."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory prediction models with optional model selection"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "transformer", "gnn", "graphidyom"],
        default=["lstm", "transformer", "gnn", "graphidyom"],
        help="Models to evaluate (space-separated). Default: all models",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["ci", "co", "cl", "mixed"],
        default=["ci", "co", "cl", "mixed"],
        help="Scenarios to evaluate (space-separated). Default: all scenarios",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "noisy"],
        default=["original", "noisy"],
        help="Dataset variants to evaluate. Default: both original and noisy",
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = base_dir / "full_path_metrics_evaluation.json"
    
    # Load existing results if available (for resuming from checkpoint)
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {output_path}")
    else:
        all_results = {}
    
    print(f"Device: {device}")
    print(f"⚠️  Note: Using CPU evaluation to avoid CUDA assertion errors on GNN models")
    device = "cpu"  # Force CPU to avoid CUDA errors with graph indexing
    
    # Try to import model classes
    try:
        from models import (
            LSTMTrajectoryPredictor,
            TransformerTrajectoryPredictor,
            GNNTrajectoryPredictor,
            GraphIDyOMPredictor,
        )
        MODELS_AVAILABLE = True
        print("[OK] Successfully imported model classes from models.py")
    except ImportError as e:
        MODELS_AVAILABLE = False
        print(f"✗ Could not import models: {e}")
        print("\nTo enable full evaluation, create a models.py with:")
        print("  - LSTMTrajectoryPredictor")
        print("  - TransformerTrajectoryPredictor")
        print("  - GNNTrajectoryPredictor")
        print("  - GraphIDyOMPredictor")
        return
    
    # Configuration
    SCENARIOS = []
    for scenario_key in args.scenarios:
        for variant in args.variants:
            SCENARIOS.append((scenario_key, variant))
    
    MODEL_CLASSES = {
        "lstm": LSTMTrajectoryPredictor,
        "transformer": TransformerTrajectoryPredictor,
        "gnn": GNNTrajectoryPredictor,
        "graphidyom": GraphIDyOMPredictor,
    }
    
    MODELS = args.models  # Use models from command-line arguments
    L_VALUES = [2, 5, 10]
    NUM_SAMPLES = 100
    K_FOLDS = 5
    
    print(f"\n{'='*70}")
    print(f"Selected models: {', '.join(MODELS)}")
    print(f"Selected scenarios: {len(SCENARIOS)} scenarios")
    print(f"{'='*70}\n")
    
    evaluator = AutoregressiveEvaluator(device=device)
    
    # Build edge_index from actual road network graph
    def build_edge_index_from_graph(graph_json_path: Path, node_to_idx: dict, device: str) -> torch.Tensor:
        """Build edge_index tensor from actual road network graph (not from sampled paths)."""
        try:
            with open(graph_json_path, 'r') as f:
                graph_data = json.load(f)
            
            total_nodes = len(graph_data.get('nodes', []))
            total_edges = len(graph_data.get('edges', []))
            vocab_size = len(node_to_idx)
            print(f"  Graph has {total_nodes} nodes, {total_edges} edges")
            print(f"  Vocabulary has {vocab_size} nodes")
            
            edges = set()
            matched_edges = 0
            
            edges_key = "links" if "links" in graph_data else "edges"
            for edge in graph_data.get(edges_key, []):
                src_id = edge.get('source')
                dst_id = edge.get('target')
                
                if src_id is not None and dst_id is not None:
                    if src_id in node_to_idx and dst_id in node_to_idx:
                        src_idx = node_to_idx[src_id]
                        dst_idx = node_to_idx[dst_id]
                        edges.add((src_idx, dst_idx))
                        # Do NOT add reverse: the JSON stores both directions explicitly
                        matched_edges += 1
            
            # Add self-loops for every node (GAT layers are trained with self-loops)
            for idx in range(len(node_to_idx)):
                edges.add((idx, idx))
            
            print(f"  Matched {matched_edges} directed edges to vocabulary nodes")
            
            if not edges:
                print(f"  ⚠️  No edges could be created (vocab mismatch)")
                return torch.zeros((2, 0), dtype=torch.long, device=device)
            
            edge_list = list(edges)
            edge_index = torch.tensor(
                [[e[0] for e in edge_list], [e[1] for e in edge_list]], 
                dtype=torch.long, device=device
            )
            print(f"  [OK] Created edge_index with {len(edges)} edges for GNN")
            return edge_index
            
        except FileNotFoundError:
            print(f"  ⚠️  Graph JSON not found at {graph_json_path}")
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        except Exception as e:
            print(f"  ⚠️  Error loading graph: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros((2, 0), dtype=torch.long, device=device)

    for scenario_key, variant in SCENARIOS:
        scenario_label = f"{scenario_key.upper()}"
        if "noisy" in variant:
            scenario_label += " (Noisy)"
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {scenario_label}")
        print(f"{'='*70}")
        
        # Determine paths and model directories based on scenario type
        if scenario_key == "mixed":
            # For mixed scenarios, combine data from CI, CO, CL
            print(f"  Loading mixed dataset (CI + CO + CL combined)")
            paths = []
            for sub_scenario in ["ci", "co", "cl"]:
                data_filename = f"subnet_{sub_scenario}" + ("_noisy" if "noisy" in variant else "") + ".csv"
                data_path = base_dir / "data" / data_filename
                
                if not data_path.exists():
                    print(f"    ⚠️  Data file not found: {data_path}")
                    continue
                
                sub_paths = load_paths_from_csv(data_path)
                print(f"    Loaded {len(sub_paths)} paths from {sub_scenario}")
                paths.extend(sub_paths)
        else:
            # Single scenario: CI, CO, or CL
            data_filename = f"subnet_{scenario_key}" + ("_noisy" if "noisy" in variant else "") + ".csv"
            data_path = base_dir / "data" / data_filename
            
            if not data_path.exists():
                print(f"⚠️  Data file not found: {data_path}")
                continue
            
            paths = load_paths_from_csv(data_path)
        print(f"Loaded {len(paths)} paths")
        
        if not paths:
            continue
        
        # graph_json_path is scenario-independent; vocab/edges are rebuilt per L below
        graph_json_path = base_dir / "data/subnet_NY.json"
        
        # Load existing results for this scenario (to preserve other models)
        if scenario_label in all_results:
            scenario_results = all_results[scenario_label].copy()
            print(f"  Loaded existing results for {scenario_label} (will preserve other models)")
        else:
            scenario_results = {}
        
        for model_name in MODELS:
            model_class = MODEL_CLASSES[model_name]
            print(f"\n  Model: {model_name.upper()}")
            
            # Load only the first fold (all folds have identical results)
            fold_idx = 0
            
            # Evaluate with different sequence lengths - LOAD MATCHING SEQUENCE LENGTH MODELS
            for L in L_VALUES:
                # CRITICAL: build vocab from paths with len >= L+1 only, matching train.py's
                # filter: self.df[self.df['route_nodes'].apply(len) >= self.sequence_length + 1]
                # Using all paths shifts node indices relative to training, corrupting predictions.
                vocab_paths = [p for p in paths if len(p) >= L + 1]
                node_to_idx, idx_to_node, vocab_size = build_node_vocab(vocab_paths)
                indexed_paths = [
                    [node_to_idx[n] for n in p]
                    for p in vocab_paths
                    if all(n in node_to_idx for n in p)
                ]
                print(f"  L={L}: vocab_size={vocab_size}, {len(indexed_paths)} valid paths")
                edge_index = build_edge_index_from_graph(graph_json_path, node_to_idx, device)
                graph_struct = evaluator.build_graph_structure_for_graphidyom(indexed_paths, vocab_size)
                
                # Load model trained on this sequence length
                if scenario_key == "mixed":
                    # Mixed datasets use different directory structure
                    if variant == "original":
                        model_base = base_dir / "model_output_original_mixed" / f"model_outputs_mixed_original_seq{L}"
                    else:
                        model_base = base_dir / "model_output_noisy_mixed" / f"model_outputs_mixed_noisy_seq{L}"
                else:
                    # Individual scenarios (CI, CO, CL)
                    if variant == "original":
                        model_base = base_dir / "model_output_original" / f"model_outputs_{scenario_key}_seq{L}"
                    else:
                        model_base = base_dir / "model_output_noisy" / f"model_outputs_noisy_{scenario_key}_seq{L}"
                
                checkpoint_path = model_base / f"{model_name}_fold_{fold_idx}_best.pth"
                
                if not checkpoint_path.exists():
                    # Try alternative naming
                    checkpoint_path = model_base / f"best_{model_name}_model_kfold.pth"
                
                if not checkpoint_path.exists():
                    print(f"    ⚠️  Checkpoint not found for L={L}: {checkpoint_path}")
                    continue
                
                print(f"    Loading checkpoint for L={L} (seq{L} model from {'mixed' if scenario_key == 'mixed' else scenario_key} dataset)")
                
                # Load model with model-specific config
                config = {"vocab_size": vocab_size}
                
                result = load_model_checkpoint(checkpoint_path, model_class, config, device)
                
                if result is None:
                    print(f"    ⚠️  Failed to load model for L={L}")
                    continue
                
                model, model_vocab_size = result
                
                # Set up evaluator for this model
                evaluator.model_type = model_name
                evaluator.vocab_size = vocab_size
                
                if model_name == "gnn":
                    evaluator.edge_index = edge_index
                    evaluator.graph_struct = None
                elif model_name == "graphidyom":
                    evaluator.edge_index = None
                    evaluator.graph_struct = graph_struct
                else:
                    evaluator.edge_index = None
                    evaluator.graph_struct = None
                
                # Evaluate with THIS sequence length
                try:
                    fold_results = evaluator.evaluate_on_paths(
                        model=model,
                        paths=indexed_paths,
                        sequence_lengths=[L],  # Only evaluate with matching L
                        num_samples=NUM_SAMPLES,
                        model_vocab_size=model_vocab_size,  # Pass model's training vocab size
                    )
                    
                    if model_name not in scenario_results:
                        scenario_results[model_name] = {}
                    
                    scenario_results[model_name][L] = {
                        "node_accuracy": float(fold_results[L]["mean_node_accuracy"]),
                        "node_accuracy_std": float(fold_results[L]["std_node_accuracy"]),
                        "normalized_edit_distance": float(fold_results[L]["mean_normalized_edit_distance"]),
                        "normalized_edit_distance_std": float(fold_results[L]["std_normalized_edit_distance"]),
                        "f1_score": float(fold_results[L]["f1_score"]),
                        "auprc": float(fold_results[L]["auprc"]),
                    }
                    
                    print(f"    L={L}: Acc={scenario_results[model_name][L]['node_accuracy']*100:.2f}%, "
                          f"EditDist={scenario_results[model_name][L]['normalized_edit_distance']:.3f}, "
                          f"F1={scenario_results[model_name][L]['f1_score']:.4f}, "
                          f"AUPRC={scenario_results[model_name][L]['auprc']:.4f}")
                    
                    # Save results incrementally after each L value
                    # Merge with existing results (preserve other models)
                    all_results[scenario_label] = scenario_results
                    with open(output_path, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"    [OK] Results saved for {scenario_label} - {model_name} - L={L}")
                    
                except Exception as e:
                    print(f"    ✗ Error during evaluation at L={L}: {e}")
                    traceback.print_exc()
                
                # Free memory
                del model
                try:
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    # CUDA may be in a bad state from failed predictions
                    # This is non-fatal - just skip cache clearing
                    print(f"    ⚠️  Could not clear CUDA cache: {str(e)[:50]}...")
                    pass
        
        # Save results after all models for this scenario
        all_results[scenario_label] = scenario_results
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"[OK] Scenario {scenario_label} completed and saved")
    
    print(f"\n{'='*70}")
    print(f"[OK] All results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    
    # Show usage information if help is requested
    if "--help" not in sys.argv and "-h" not in sys.argv:
        print("\n" + "="*70)
        print("USAGE EXAMPLES:")
        print("="*70)
        print("# Run all models on all scenarios:")
        print("  python evaluate_full_path_prediction_v2.py")
        print("\n# Run only specific models:")
        print("  python evaluate_full_path_prediction_v2.py --models lstm transformer")
        print("\n# Run GNN model only:")
        print("  python evaluate_full_path_prediction_v2.py --models gnn")
        print("\n# Run specific scenarios:")
        print("  python evaluate_full_path_prediction_v2.py --scenarios ci co")
        print("\n# Run only original datasets (no noisy):")
        print("  python evaluate_full_path_prediction_v2.py --variants original")
        print("\n# Combine options:")
        print("  python evaluate_full_path_prediction_v2.py --models lstm gnn --scenarios ci")
        print("="*70 + "\n")
    
    main()
