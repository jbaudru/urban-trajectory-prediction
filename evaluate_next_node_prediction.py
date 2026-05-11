"""
Evaluate next-node prediction metrics (non-autoregressive / single-step).

This script expects a `models.py` module in the same directory with:
- LSTMTrajectoryPredictor
- TransformerTrajectoryPredictor
- GNNTrajectoryPredictor
- GraphIDyOMPredictor

Evaluation methodology (mirrors train.py's sliding-window dataset):
  For each path of length N and context window L:
    - Slide a window of length L across the path
    - At each position i, context = path[i:i+L], target = path[i+L]
    - Perform a single forward pass to get logits / softmax probs
    - Collect (true_node, predicted_node, prob_distribution) for all windows

Metrics computed per (scenario, model, L):
  - top1_accuracy          : fraction of steps where argmax == true node
  - top1_accuracy_std      : std across sampled paths
  - f1_score               : weighted F1 across all prediction steps
  - auprc                  : mean per-step average precision (true class vs rest)

Output: next_node_metrics_evaluation.json
"""

from __future__ import annotations

import json
import csv
import argparse
from pathlib import Path
from typing import Optional
import random
import traceback

import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths_from_csv(csv_path: Path) -> list[list[int]]:
    """Load paths from CSV file (q_path column, comma-separated node IDs)."""
    paths = []
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "q_path" in row and row["q_path"].strip():
                try:
                    path = [int(n) for n in row["q_path"].split(",")]
                    if len(path) >= 2:
                        paths.append(path)
                except (ValueError, KeyError):
                    continue
    return paths


def build_node_vocab(
    paths: list[list[int]],
) -> tuple[dict[int, int], dict[int, int], int]:
    """Build node-to-index and index-to-node mappings from a collection of paths."""
    unique_nodes: set[int] = set()
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
    """Load model weights from a checkpoint file.

    Returns:
        (model, checkpoint_vocab_size) or None on failure.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )

        checkpoint_vocab_size: Optional[int] = None
        checkpoint_embedding_dim: Optional[int] = None

        for layer_name in [
            "embedding.weight",
            "node_embedding.weight",
            "token_embedding.weight",
        ]:
            if layer_name in state_dict:
                checkpoint_vocab_size = state_dict[layer_name].shape[0]
                checkpoint_embedding_dim = state_dict[layer_name].shape[1]
                print(
                    f"      Inferred vocab_size={checkpoint_vocab_size}, "
                    f"embedding_dim={checkpoint_embedding_dim} from '{layer_name}'"
                )
                config["vocab_size"] = checkpoint_vocab_size
                config["embedding_dim"] = checkpoint_embedding_dim
                break

        if checkpoint_vocab_size is None:
            for layer_name in ["classifier.weight", "fc.weight"]:
                if layer_name in state_dict:
                    checkpoint_vocab_size = state_dict[layer_name].shape[0]
                    print(
                        f"      Inferred vocab_size={checkpoint_vocab_size} from '{layer_name}'"
                    )
                    config["vocab_size"] = checkpoint_vocab_size
                    break

        if checkpoint_vocab_size is None:
            checkpoint_vocab_size = config.get("vocab_size")

        try:
            model = model_class(**config)
        except TypeError:
            config.pop("embedding_dim", None)
            model = model_class(**config)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, checkpoint_vocab_size

    except Exception as exc:
        print(f"Error loading model from {checkpoint_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class NextNodeEvaluator:
    """Single-step (non-autoregressive) next-node prediction evaluator."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_type: Optional[str] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.graph_struct: Optional[dict] = None
        self.vocab_size: Optional[int] = None
        self.model_vocab_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Single forward pass returning probs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_probs(
        self, model: torch.nn.Module, context: torch.Tensor
    ) -> tuple[int, np.ndarray]:
        """Run one forward pass and return (argmax_node, softmax_probs)."""
        context = context.to(self.device)

        if (
            self.model_vocab_size is not None
            and context.max().item() >= self.model_vocab_size
        ):
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

        logits = output[0] if isinstance(output, tuple) else output

        if logits.dim() == 3:
            last_logits = logits[0, -1, :]
        elif logits.dim() == 2:
            last_logits = logits[0, :]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        probs = torch.softmax(last_logits, dim=-1).cpu().numpy()
        pred_node = int(np.argmax(probs))

        if self.model_vocab_size is not None and pred_node >= self.model_vocab_size:
            pred_node = self.model_vocab_size - 1

        return pred_node, probs

    # ------------------------------------------------------------------
    # Graph structure builder (for GraphIDyOM)
    # ------------------------------------------------------------------

    def build_graph_structure_for_graphidyom(
        self, paths: list[list[int]], vocab_size: int
    ) -> dict:
        """Build adjacency/degree tensors from path transitions."""
        print(
            f"    Building graph structure from {len(paths)} paths "
            f"with vocab_size={vocab_size}"
        )
        adjacency = torch.zeros(
            (vocab_size, vocab_size), device=self.device, dtype=torch.float32
        )
        for path in paths:
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                if src < vocab_size and dst < vocab_size:
                    adjacency[src, dst] += 1.0

        row_sums = adjacency.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        adjacency_normalized = adjacency / row_sums

        in_degrees = adjacency.sum(dim=0)
        out_degrees = adjacency.sum(dim=1)
        print(
            f"    [OK] Graph: {adjacency.count_nonzero().item()} non-zero adjacency entries"
        )
        return {
            "adjacency": adjacency_normalized,
            "edge_weights": adjacency_normalized,
            "in_degrees": in_degrees,
            "out_degrees": out_degrees,
        }

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_on_paths(
        self,
        model: torch.nn.Module,
        paths: list[list[int]],
        sequence_lengths: list[int] = [2, 5, 10],
        num_samples: int = 500,
        seed: int = 42,
        model_vocab_size: Optional[int] = None,
    ) -> dict[int, dict[str, float]]:
        """Evaluate next-node prediction for each context length L.

        For each L:
          1. Build all (context, target) windows from the sampled paths.
          2. Single forward pass per window.
          3. Aggregate accuracy, weighted F1, and AUPRC.

        Args:
            model: Trained model in eval mode.
            paths: Indexed paths (node indices already remapped to [0, vocab_size)).
            sequence_lengths: Context window sizes to evaluate.
            num_samples: Maximum number of *paths* to sample (windows are derived from them).
            seed: Random seed for path sampling.
            model_vocab_size: Vocab size the model was trained on.

        Returns:
            Dict mapping L -> metric dict.
        """
        random.seed(seed)
        self.model_vocab_size = model_vocab_size
        results: dict[int, dict[str, float]] = {}

        # Filter edge_index if model has a smaller vocab than the test set
        if (
            model_vocab_size is not None
            and self.vocab_size is not None
            and model_vocab_size < self.vocab_size
            and self.edge_index is not None
            and self.edge_index.numel() > 0
        ):
            valid = (self.edge_index[0] < model_vocab_size) & (
                self.edge_index[1] < model_vocab_size
            )
            self.edge_index = self.edge_index[:, valid]
            print(
                f"  Filtered edge_index: kept {valid.sum().item()} edges "
                f"(model vocab {model_vocab_size})"
            )

        # Filter paths whose nodes exceed model vocab
        valid_paths = paths
        if model_vocab_size is not None and self.vocab_size is not None and model_vocab_size < self.vocab_size:
            valid_paths = [
                p for p in paths if all(n < model_vocab_size for n in p)
            ]
            skipped = len(paths) - len(valid_paths)
            if skipped:
                print(
                    f"  Filtered {skipped} paths with nodes outside model vocab "
                    f"(>= {model_vocab_size})"
                )

        sampled_paths = random.sample(valid_paths, min(num_samples, len(valid_paths)))
        print(f"  Sampled {len(sampled_paths)} paths")

        for L in sequence_lengths:
            # Build sliding-window dataset
            windows: list[tuple[list[int], int]] = []  # (context, true_next_node)
            for path in sampled_paths:
                if len(path) <= L:
                    continue
                for i in range(len(path) - L):
                    context = path[i : i + L]
                    target = path[i + L]
                    windows.append((context, target))

            if not windows:
                results[L] = {
                    "top1_accuracy": 0.0,
                    "top1_accuracy_std": 0.0,
                    "f1_score": 0.0,
                    "auprc": 0.0,
                    "num_samples": 0,
                }
                print(f"    L={L}: no valid windows")
                continue

            print(f"    L={L}: {len(windows)} windows from {len(sampled_paths)} paths")

            per_path_accs: list[float] = []
            all_true: list[int] = []
            all_pred: list[int] = []
            all_step_aps: list[float] = []

            # Group windows back by path for per-path accuracy
            path_windows: list[list[tuple[list[int], int]]] = []
            for path in sampled_paths:
                if len(path) <= L:
                    continue
                pw = [
                    (path[i : i + L], path[i + L])
                    for i in range(len(path) - L)
                ]
                path_windows.append(pw)

            for pw in path_windows:
                path_correct = 0
                for context, true_node in pw:
                    ctx_tensor = torch.tensor([context], dtype=torch.long)
                    try:
                        pred_node, probs = self._forward_probs(model, ctx_tensor)
                    except Exception as exc:
                        print(f"      Prediction error: {exc}")
                        continue

                    all_true.append(true_node)
                    all_pred.append(pred_node)
                    if pred_node == true_node:
                        path_correct += 1

                    # Per-step AP
                    prob_len = len(probs)
                    binary_true = np.zeros(prob_len, dtype=np.int32)
                    if true_node < prob_len:
                        binary_true[true_node] = 1
                    if binary_true.sum() > 0:
                        try:
                            ap = average_precision_score(binary_true, probs)
                            all_step_aps.append(float(ap))
                        except ValueError:
                            pass

                if pw:
                    per_path_accs.append(path_correct / len(pw))

            # Aggregate
            top1_acc = float(np.mean(per_path_accs)) if per_path_accs else 0.0
            top1_std = float(np.std(per_path_accs)) if per_path_accs else 0.0

            f1 = (
                float(
                    f1_score(all_true, all_pred, average="weighted", zero_division=0)
                )
                if all_true
                else 0.0
            )
            auprc = float(np.mean(all_step_aps)) if all_step_aps else 0.0

            results[L] = {
                "top1_accuracy": top1_acc,
                "top1_accuracy_std": top1_std,
                "f1_score": f1,
                "auprc": auprc,
                "num_samples": len(per_path_accs),
            }

            print(
                f"    L={L}: {len(per_path_accs)} paths, {len(all_true)} windows | "
                f"Acc={top1_acc:.4f}, F1={f1:.4f}, AUPRC={auprc:.4f}"
            )

        return results


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_edge_index_from_graph(
    graph_json_path: Path, node_to_idx: dict, device: str
) -> torch.Tensor:
    """Build edge_index tensor from road network JSON."""
    try:
        with open(graph_json_path, "r") as f:
            graph_data = json.load(f)

        print(
            f"  Graph: {len(graph_data.get('nodes', []))} nodes, "
            f"{len(graph_data.get('edges', graph_data.get('links', [])))} edges"
        )

        edges: set[tuple[int, int]] = set()
        edges_key = "links" if "links" in graph_data else "edges"
        matched = 0
        for edge in graph_data.get(edges_key, []):
            src_id = edge.get("source")
            dst_id = edge.get("target")
            if src_id in node_to_idx and dst_id in node_to_idx:
                edges.add((node_to_idx[src_id], node_to_idx[dst_id]))
                matched += 1

        # Self-loops (required by GAT layers)
        for idx in range(len(node_to_idx)):
            edges.add((idx, idx))

        print(f"  Matched {matched} edges; total with self-loops: {len(edges)}")

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        edge_list = list(edges)
        return torch.tensor(
            [[e[0] for e in edge_list], [e[1] for e in edge_list]],
            dtype=torch.long,
            device=device,
        )
    except FileNotFoundError:
        print(f"  Graph JSON not found: {graph_json_path}")
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    except Exception as exc:
        print(f"  Error loading graph: {exc}")
        traceback.print_exc()
        return torch.zeros((2, 0), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate next-node (single-step) prediction metrics"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "transformer", "gnn", "graphidyom"],
        default=["lstm", "transformer", "gnn", "graphidyom"],
        help="Models to evaluate. Default: all.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["ci", "co", "cl", "mixed"],
        default=["ci", "co", "cl", "mixed"],
        help="Scenarios to evaluate. Default: all.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "noisy"],
        default=["original", "noisy"],
        help="Dataset variants. Default: both.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Max number of paths to sample per (scenario, L). Default: 500.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "next_node_metrics_evaluation.json"
    device = "cpu"  # Force CPU to avoid CUDA graph-indexing errors

    # Load existing results (resume support)
    if output_path.exists():
        with open(output_path, "r") as f:
            all_results: dict = json.load(f)
        print(f"Loaded existing results from {output_path}")
    else:
        all_results = {}

    # Import model classes
    try:
        from models import (
            LSTMTrajectoryPredictor,
            TransformerTrajectoryPredictor,
            GNNTrajectoryPredictor,
            GraphIDyOMPredictor,
        )
        print("[OK] Imported model classes from models.py")
    except ImportError as exc:
        print(f"Could not import models: {exc}")
        return

    MODEL_CLASSES = {
        "lstm": LSTMTrajectoryPredictor,
        "transformer": TransformerTrajectoryPredictor,
        "gnn": GNNTrajectoryPredictor,
        "graphidyom": GraphIDyOMPredictor,
    }
    MODELS = args.models
    L_VALUES = [2, 5, 10]
    NUM_SAMPLES = args.num_samples

    SCENARIOS = [
        (scenario_key, variant)
        for scenario_key in args.scenarios
        for variant in args.variants
    ]

    print(f"\n{'='*70}")
    print(f"Next-node (single-step) evaluation")
    print(f"Models   : {', '.join(MODELS)}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Device   : {device}")
    print(f"{'='*70}\n")

    evaluator = NextNodeEvaluator(device=device)
    graph_json_path = base_dir / "data" / "subnet_NY.json"

    for scenario_key, variant in SCENARIOS:
        scenario_label = scenario_key.upper()
        if "noisy" in variant:
            scenario_label += " (Noisy)"

        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_label}")
        print(f"{'='*70}")

        # Load raw paths
        if scenario_key == "mixed":
            paths: list[list[int]] = []
            for sub in ["ci", "co", "cl"]:
                fname = f"subnet_{sub}" + ("_noisy" if "noisy" in variant else "") + ".csv"
                dp = base_dir / "data" / fname
                if not dp.exists():
                    print(f"  Data file not found: {dp}")
                    continue
                sub_paths = load_paths_from_csv(dp)
                print(f"  Loaded {len(sub_paths)} paths from {sub}")
                paths.extend(sub_paths)
        else:
            fname = (
                f"subnet_{scenario_key}"
                + ("_noisy" if "noisy" in variant else "")
                + ".csv"
            )
            dp = base_dir / "data" / fname
            if not dp.exists():
                print(f"  Data file not found: {dp}")
                continue
            paths = load_paths_from_csv(dp)

        print(f"  Total paths loaded: {len(paths)}")
        if not paths:
            continue

        # Preserve existing results for this scenario
        scenario_results: dict = (
            all_results[scenario_label].copy()
            if scenario_label in all_results
            else {}
        )

        for model_name in MODELS:
            model_class = MODEL_CLASSES[model_name]
            print(f"\n  Model: {model_name.upper()}")
            fold_idx = 0

            for L in L_VALUES:
                # Build vocab from paths long enough for this L (mirrors train.py filtering)
                vocab_paths = [p for p in paths if len(p) >= L + 1]
                node_to_idx, _, vocab_size = build_node_vocab(vocab_paths)
                indexed_paths = [
                    [node_to_idx[n] for n in p]
                    for p in vocab_paths
                    if all(n in node_to_idx for n in p)
                ]
                print(f"  L={L}: vocab_size={vocab_size}, {len(indexed_paths)} valid paths")

                edge_index = build_edge_index_from_graph(graph_json_path, node_to_idx, device)
                graph_struct = evaluator.build_graph_structure_for_graphidyom(
                    indexed_paths, vocab_size
                )

                # Locate checkpoint
                if scenario_key == "mixed":
                    model_base = (
                        base_dir / "model_output_original_mixed"
                        / f"model_outputs_mixed_original_seq{L}"
                        if variant == "original"
                        else base_dir / "model_output_noisy_mixed"
                        / f"model_outputs_mixed_noisy_seq{L}"
                    )
                else:
                    model_base = (
                        base_dir / "model_output_original"
                        / f"model_outputs_{scenario_key}_seq{L}"
                        if variant == "original"
                        else base_dir / "model_output_noisy"
                        / f"model_outputs_noisy_{scenario_key}_seq{L}"
                    )

                checkpoint_path = model_base / f"{model_name}_fold_{fold_idx}_best.pth"
                if not checkpoint_path.exists():
                    checkpoint_path = model_base / f"best_{model_name}_model_kfold.pth"
                if not checkpoint_path.exists():
                    print(f"    Checkpoint not found: {checkpoint_path}")
                    continue

                print(f"    Loading checkpoint: {checkpoint_path.name}")
                config = {"vocab_size": vocab_size}
                result = load_model_checkpoint(checkpoint_path, model_class, config, device)
                if result is None:
                    print(f"    Failed to load checkpoint for L={L}")
                    continue

                model, model_vocab_size = result

                # Configure evaluator
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

                try:
                    fold_results = evaluator.evaluate_on_paths(
                        model=model,
                        paths=indexed_paths,
                        sequence_lengths=[L],
                        num_samples=NUM_SAMPLES,
                        model_vocab_size=model_vocab_size,
                    )

                    if model_name not in scenario_results:
                        scenario_results[model_name] = {}

                    scenario_results[model_name][L] = {
                        "top1_accuracy": float(fold_results[L]["top1_accuracy"]),
                        "top1_accuracy_std": float(fold_results[L]["top1_accuracy_std"]),
                        "f1_score": float(fold_results[L]["f1_score"]),
                        "auprc": float(fold_results[L]["auprc"]),
                        "num_samples": int(fold_results[L]["num_samples"]),
                    }

                    r = scenario_results[model_name][L]
                    print(
                        f"    L={L}: Acc={r['top1_accuracy']*100:.2f}%, "
                        f"F1={r['f1_score']:.4f}, AUPRC={r['auprc']:.4f}"
                    )

                    # Incremental save
                    all_results[scenario_label] = scenario_results
                    with open(output_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    print(f"    [OK] Saved {scenario_label} / {model_name} / L={L}")

                except Exception as exc:
                    print(f"    Error at L={L}: {exc}")
                    traceback.print_exc()

                del model
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass

        all_results[scenario_label] = scenario_results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[OK] Scenario {scenario_label} complete")

    print(f"\n{'='*70}")
    print(f"[OK] Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys

    if "--help" not in sys.argv and "-h" not in sys.argv:
        print("\n" + "=" * 70)
        print("USAGE EXAMPLES:")
        print("=" * 70)
        print("  python evaluate_next_node_prediction.py")
        print("  python evaluate_next_node_prediction.py --models lstm transformer")
        print("  python evaluate_next_node_prediction.py --scenarios ci co --variants original")
        print("  python evaluate_next_node_prediction.py --num-samples 200")
        print("=" * 70 + "\n")

    main()
