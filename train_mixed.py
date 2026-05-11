"""
Training script for mixed-scenario trajectory prediction.
Trains LSTM, Transformer, and GNN models on a balanced mix of CI/CO/CL datasets.
"""

import argparse
import ast
import os
from datetime import datetime

import numpy as np
import pandas as pd

from train import (
    TrajectoryPredictor,
    TRAIN_GNN,
    TRAIN_LSTM,
    TRAIN_TRANSFORMER,
    TRAIN_GRAPHIDYOM,
    K_FOLDS,
    USE_KFOLD,
)


class MixedTrajectoryPredictor(TrajectoryPredictor):
    """Trajectory predictor that loads a balanced CI/CO/CL mixed dataset."""

    def __init__(
        self,
        data_ci_path,
        data_co_path,
        data_cl_path,
        graph_path,
        sequence_length=10,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    ):
        self.data_paths = {
            "ci": data_ci_path,
            "co": data_co_path,
            "cl": data_cl_path,
        }
        self.random_state = random_state
        super().__init__(
            data_path="MIXED_DATASET",
            graph_path=graph_path,
            sequence_length=sequence_length,
            test_size=test_size,
            val_size=val_size,
        )

    def _load_single_dataset(self, path):
        df = pd.read_csv(path)
        if "q_path" in df.columns:
            def parse_q_path(path_str):
                if isinstance(path_str, str):
                    return [int(node.strip()) for node in path_str.split(",") if node.strip()]
                return []
            df["route_nodes"] = df["q_path"].apply(parse_q_path)
        elif "route_taken" in df.columns:
            df["route_nodes"] = df["route_taken"].apply(ast.literal_eval)
        else:
            raise ValueError(f"No valid path column in {path}. Expected 'q_path' or 'route_taken'.")

        df = df[df["route_nodes"].apply(len) >= self.sequence_length + 1].copy()
        return df

    def load_data(self):
        """Load CI/CO/CL datasets and build a strict 33/33/33 mixed dataset."""
        print("Loading mixed trajectory data (CI/CO/CL)...")

        dataframes = {}
        for key, path in self.data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found for {key.upper()}: {path}")
            df = self._load_single_dataset(path)
            dataframes[key] = df
            print(f"  {key.upper()}: {len(df)} valid trips")

        # Strict balanced mix: same number of trips from each scenario.
        n_per_dataset = min(len(dataframes["ci"]), len(dataframes["co"]), len(dataframes["cl"]))
        if n_per_dataset == 0:
            raise ValueError("At least one dataset has 0 valid trips after filtering.")

        mixed_parts = []
        for key in ["ci", "co", "cl"]:
            sampled = dataframes[key].sample(n=n_per_dataset, random_state=self.random_state)
            sampled = sampled.copy()
            sampled["scenario"] = key
            mixed_parts.append(sampled)

        self.df = pd.concat(mixed_parts, ignore_index=True)
        self.df = self.df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        print(f"Balanced mixed dataset: {len(self.df)} trips")
        print(f"  Per scenario: {n_per_dataset} ({100/3:.2f}% each)")


def main():
    parser = argparse.ArgumentParser(description="Train trajectory models on mixed CI/CO/CL data")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "gnn", "graphidyom", "all"], default="all",
                        help="Model type to train (default: all)")

    parser.add_argument("--data-ci-path", type=str, default="data/subnet_ci.csv",
                        help="Path to CI dataset CSV")
    parser.add_argument("--data-co-path", type=str, default="data/subnet_co.csv",
                        help="Path to CO dataset CSV")
    parser.add_argument("--data-cl-path", type=str, default="data/subnet_cl.csv",
                        help="Path to CL dataset CSV")

    parser.add_argument("--graph-path", type=str, default="data/subnet_NY.json",
                        help="Path to graph JSON file")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--sequence-length", type=int, default=10,
                        help="Sequence length (user choice)")
    parser.add_argument("--output-dir", type=str, default="model_outputs_mixed",
                        help="Output directory for models and results")

    parser.add_argument("--kfold", action="store_true", default=None,
                        help="Use K-Fold cross-validation")
    parser.add_argument("--no-kfold", action="store_true", default=None,
                        help="Disable K-Fold cross-validation")
    parser.add_argument("--k-folds", type=int, default=None,
                        help=f"Number of folds for K-Fold CV (default: {K_FOLDS})")
    parser.add_argument("--graphidyom-order", type=int, default=7,
                        help="Markov order for GraphIDyOM model (default: 7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for balanced mixing and shuffling")

    args = parser.parse_args()

    use_kfold = USE_KFOLD
    if args.kfold:
        use_kfold = True
    elif args.no_kfold:
        use_kfold = False
    k_folds = args.k_folds if args.k_folds is not None else K_FOLDS

    required_paths = [args.data_ci_path, args.data_co_path, args.data_cl_path, args.graph_path]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: File not found: {p}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    predictor = MixedTrajectoryPredictor(
        data_ci_path=f"../{args.data_ci_path}",
        data_co_path=f"../{args.data_co_path}",
        data_cl_path=f"../{args.data_cl_path}",
        graph_path=f"../{args.graph_path}",
        sequence_length=args.sequence_length,
        test_size=0.2,
        val_size=0.2,
        random_state=args.seed,
    )

    if args.model == "all":
        print("\n" + "=" * 60)
        print("MIXED DATA TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"  Dataset mix: CI/CO/CL = 33/33/33")
        print(f"  Sequence length: {args.sequence_length}")
        print(f"  TRAIN_LSTM: {TRAIN_LSTM}")
        print(f"  TRAIN_TRANSFORMER: {TRAIN_TRANSFORMER}")
        print(f"  TRAIN_GNN: {TRAIN_GNN}")
        print(f"  TRAIN_GRAPHIDYOM: {TRAIN_GRAPHIDYOM}")
        print(f"  USE_KFOLD: {use_kfold}")
        if use_kfold:
            print(f"  K_FOLDS: {k_folds}")

        results, _ = predictor.compare_models(
            train_lstm=TRAIN_LSTM,
            train_transformer=TRAIN_TRANSFORMER,
            train_gnn=TRAIN_GNN,
            train_graphidyom=TRAIN_GRAPHIDYOM,
            use_kfold=use_kfold,
            k_folds=k_folds,
            graphidyom_order=args.graphidyom_order,
        )

        if results:
            print("\n" + "=" * 60)
            print("FINAL MIXED-DATA RESULTS")
            print("=" * 60)
            for model_type, result in results.items():
                if use_kfold:
                    m = result["aggregated_metrics"]
                    print(f"{model_type.upper()}: Test Accuracy = {m['mean_test_accuracy']:.2f}% (±{m['std_test_accuracy']:.2f}%)")
                else:
                    print(f"{model_type.upper()}: Test Accuracy = {result['accuracy']:.4f}")
    else:
        print("\n" + "=" * 50)
        print(f"Training {args.model.upper()} on MIXED data")
        print(f"Sequence length: {args.sequence_length}")
        print("=" * 50)

        model_configs = {
            "lstm": {"batch_size": args.batch_size, "epochs": args.epochs, "learning_rate": args.learning_rate},
            "transformer": {
                "batch_size": min(args.batch_size, 32),
                "epochs": args.epochs,
                "learning_rate": max(args.learning_rate * 0.1, 0.0001),
            },
            "gnn": {
                "batch_size": min(args.batch_size, 32),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
            "graphidyom": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
        }
        config = model_configs[args.model]

        if use_kfold:
            _, model, aggregated_metrics = predictor.train_model_kfold(
                model_type=args.model,
                k_folds=k_folds,
                graphidyom_order=args.graphidyom_order,
                **config,
            )
            full_path_results = predictor.evaluate_full_path(model, args.model, num_samples=100)
            print(f"\n{args.model.upper()} Final Results ({k_folds}-Fold CV):")
            print(f"Test Accuracy: {aggregated_metrics['mean_test_accuracy']:.2f}% (±{aggregated_metrics['std_test_accuracy']:.2f}%)")
            if "mean_node_accuracy" in full_path_results:
                print(f"Node Accuracy: {full_path_results['mean_node_accuracy']:.4f}")
        else:
            model, train_losses, val_losses, val_accuracies = predictor.train_model(
                model_type=args.model,
                graphidyom_order=args.graphidyom_order,
                **config,
            )
            accuracy, _, _ = predictor.evaluate_model(model, args.model)
            full_path_results = predictor.evaluate_full_path(model, args.model, num_samples=100)
            print(f"\n{args.model.upper()} Final Results:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
            print(f"Final Training Loss: {train_losses[-1]:.4f}")
            print(f"Final Validation Loss: {val_losses[-1]:.4f}")
            if "mean_node_accuracy" in full_path_results:
                print(f"Node Accuracy: {full_path_results['mean_node_accuracy']:.4f}")

    print(f"\nAll outputs saved in '{args.output_dir}'")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
