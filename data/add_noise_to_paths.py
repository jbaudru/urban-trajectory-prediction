"""
Script to add noise (detours) to user paths in the CI, CO, and CL datasets.

For each user trip, we introduce random detours by:
1. Selecting random intermediate segments along the original shortest path
2. Penalizing the original edges and computing an alternative sub-path
3. Replacing the original segment with the detour

The resulting paths are still valid paths on the road network, but are no longer
the shortest paths — they contain realistic detours.

Usage:
    python add_noise_to_paths.py [--noise-prob 0.4] [--penalty 5.0] [--max-detours 3] [--seed 42]
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
import argparse
import os
import copy
from pathlib import Path


def load_network(json_path: str) -> nx.DiGraph:
    """Load the road network from the JSON file into a NetworkX DiGraph."""
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["id"], x=node["x"], y=node["y"])

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            length=edge["length"],
            speed_kph=edge.get("speed_kph", 30),
            travel_time=edge.get("travel_time", edge["length"] / 30),
        )

    return G


def parse_path(path_str: str) -> list:
    """Parse a comma-separated path string into a list of node IDs."""
    if pd.isna(path_str) or path_str.strip() == "":
        return []
    # Remove surrounding quotes if present
    path_str = path_str.strip().strip('"')
    return [int(n.strip()) for n in path_str.split(",") if n.strip()]


def path_to_string(path: list) -> str:
    """Convert a list of node IDs back to the CSV path format."""
    return ",".join(str(n) for n in path)


def compute_path_length(G: nx.DiGraph, path: list) -> float:
    """Compute the total length (in meters) of a path on the graph."""
    total = 0.0
    for i in range(len(path) - 1):
        if G.has_edge(path[i], path[i + 1]):
            total += G[path[i]][path[i + 1]]["length"]
        else:
            # Edge missing — shouldn't happen for valid paths
            total += 0
    return total


def add_detour_to_path(
    G: nx.DiGraph,
    original_path: list,
    noise_prob: float = 0.4,
    penalty_factor: float = 5.0,
    max_detours: int = 3,
    rng: np.random.Generator = None,
) -> list:
    """
    Add detours to an original path by re-routing random sub-segments.

    Parameters
    ----------
    G : nx.DiGraph
        The road network graph.
    original_path : list
        Original list of node IDs (shortest path).
    noise_prob : float
        Probability of attempting a detour at each candidate segment.
    penalty_factor : float
        Multiplicative penalty applied to original-path edges when
        computing the alternative sub-path (higher = bigger detour).
    max_detours : int
        Maximum number of detour segments to introduce.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list
        The noisy path (still a valid path on G).
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(original_path) < 4:
        # Path too short for meaningful detours
        return list(original_path)

    path = list(original_path)
    original_edges = set(zip(original_path[:-1], original_path[1:]))

    # Create a penalized copy of the graph for alternative routing
    G_penalized = G.copy()
    for u, v in original_edges:
        if G_penalized.has_edge(u, v):
            G_penalized[u][v]["length"] *= penalty_factor

    # Choose candidate split points for detour segments
    # We pick pairs of indices that are 3-15 hops apart
    n = len(path)
    min_segment = min(3, n // 2)
    max_segment = min(15, n - 1)

    detour_count = 0
    attempts = 0
    max_attempts = max_detours * 3

    # Track which indices have been modified to avoid overlapping detours
    modified_ranges = []

    while detour_count < max_detours and attempts < max_attempts:
        attempts += 1

        if rng.random() > noise_prob:
            continue

        # Pick a random start index and segment length
        seg_len = rng.integers(min_segment, max_segment + 1)
        max_start = len(path) - seg_len - 1
        if max_start < 0:
            continue
        start_idx = rng.integers(0, max_start + 1)
        end_idx = start_idx + seg_len

        # Check for overlap with existing detours
        overlaps = False
        for ms, me in modified_ranges:
            if start_idx < me and end_idx > ms:
                overlaps = True
                break
        if overlaps:
            continue

        src_node = path[start_idx]
        dst_node = path[end_idx]

        # Try to find an alternative path on the penalized graph
        try:
            alt_subpath = nx.shortest_path(
                G_penalized, src_node, dst_node, weight="length"
            )
        except nx.NetworkXNoPath:
            continue

        # Only accept if the alternative is actually different and not too long
        orig_subpath = path[start_idx : end_idx + 1]
        if alt_subpath == orig_subpath:
            continue

        # Check detour isn't excessively long (max 3x original segment)
        orig_len = compute_path_length(G, orig_subpath)
        alt_len = compute_path_length(G, alt_subpath)
        if orig_len > 0 and alt_len / orig_len > 3.0:
            continue

        # Verify all edges in the alternative path exist in the original graph
        valid = True
        for i in range(len(alt_subpath) - 1):
            if not G.has_edge(alt_subpath[i], alt_subpath[i + 1]):
                valid = False
                break
        if not valid:
            continue

        # Replace the segment
        new_path = path[:start_idx] + alt_subpath + path[end_idx + 1 :]
        path = new_path
        modified_ranges.append((start_idx, start_idx + len(alt_subpath) - 1))
        detour_count += 1

    return path


def process_dataset(
    G: nx.DiGraph,
    input_csv: str,
    output_csv: str,
    noise_prob: float,
    penalty_factor: float,
    max_detours: int,
    seed: int,
):
    """Process a single CSV dataset, adding noise to all paths."""
    print(f"\nProcessing: {input_csv}")
    df = pd.read_csv(input_csv)

    rng = np.random.default_rng(seed)
    noisy_paths = []
    noisy_distances = []
    stats = {"total": 0, "modified": 0, "unchanged": 0, "errors": 0}

    for idx, row in df.iterrows():
        stats["total"] += 1
        original_path = parse_path(str(row["q_path"]))

        if len(original_path) < 2:
            noisy_paths.append(row["q_path"])
            noisy_distances.append(row["q_km"])
            stats["unchanged"] += 1
            continue

        try:
            noisy_path = add_detour_to_path(
                G,
                original_path,
                noise_prob=noise_prob,
                penalty_factor=penalty_factor,
                max_detours=max_detours,
                rng=rng,
            )

            if noisy_path != original_path:
                stats["modified"] += 1
            else:
                stats["unchanged"] += 1

            noisy_paths.append(path_to_string(noisy_path))
            noisy_km = compute_path_length(G, noisy_path) / 1000.0  # m -> km
            # Use the computed noisy distance, but keep at least the original
            # distance as a sanity floor (graph precision may differ slightly)
            noisy_distances.append(round(noisy_km, 3))

        except Exception as e:
            # If anything goes wrong, keep the original path
            noisy_paths.append(row["q_path"])
            noisy_distances.append(row["q_km"])
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  Warning: Error on row {idx} ({row['user_id']}): {e}")

    df["q_path"] = noisy_paths
    df["q_km"] = noisy_distances
    df.to_csv(output_csv, index=False)

    print(f"  Total trips: {stats['total']}")
    print(f"  Modified:    {stats['modified']} ({100*stats['modified']/max(stats['total'],1):.1f}%)")
    print(f"  Unchanged:   {stats['unchanged']}")
    print(f"  Errors:      {stats['errors']}")
    print(f"  Saved to:    {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Add noise (detours) to shortest-path trajectories"
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.4,
        help="Probability of attempting a detour per candidate segment (default: 0.4)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=5.0,
        help="Penalty factor on original edges when computing detours (default: 5.0)",
    )
    parser.add_argument(
        "--max-detours",
        type=int,
        default=3,
        help="Maximum number of detour segments per path (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data files (default: ./data)",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"

    network_path = data_dir / "subnet_NY.json"
    datasets = {
        "subnet_ci.csv": "subnet_ci_noisy.csv",
        "subnet_co.csv": "subnet_co_noisy.csv",
        "subnet_cl.csv": "subnet_cl_noisy.csv",
    }

    # Load the road network
    print("=" * 60)
    print("ADDING NOISE TO SHORTEST-PATH TRAJECTORIES")
    print("=" * 60)
    print(f"  Noise probability : {args.noise_prob}")
    print(f"  Penalty factor    : {args.penalty}")
    print(f"  Max detours/path  : {args.max_detours}")
    print(f"  Seed              : {args.seed}")
    print(f"  Data directory    : {data_dir}")
    print()

    print("Loading road network...")
    G = load_network(str(network_path))
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Process each dataset
    for input_name, output_name in datasets.items():
        input_path = data_dir / input_name
        output_path = data_dir / output_name
        if input_path.exists():
            process_dataset(
                G,
                str(input_path),
                str(output_path),
                noise_prob=args.noise_prob,
                penalty_factor=args.penalty,
                max_detours=args.max_detours,
                seed=args.seed,
            )
        else:
            print(f"\nSkipping {input_name} (not found)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
