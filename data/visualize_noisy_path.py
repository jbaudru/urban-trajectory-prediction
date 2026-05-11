"""
Visualize the original (shortest) path vs. the noisy (detoured) path
for a given user, overlaid on the road network.

Usage:
    python visualize_noisy_path.py [--user-id p_81] [--dataset ci] [--save]
    python visualize_noisy_path.py --random --dataset co
"""

import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import argparse
from pathlib import Path

# IEEE paper style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
})


def load_network(json_path: str):
    """Load the road network and return a NetworkX DiGraph + node positions."""
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    pos = {}

    for node in data["nodes"]:
        nid = node["id"]
        G.add_node(nid, x=node["x"], y=node["y"])
        pos[nid] = (node["x"], node["y"])

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            length=edge["length"],
        )

    return G, pos


def parse_path(path_str: str) -> list:
    """Parse a comma-separated path string into a list of node IDs."""
    if pd.isna(path_str) or str(path_str).strip() == "":
        return []
    path_str = str(path_str).strip().strip('"')
    return [int(n.strip()) for n in path_str.split(",") if n.strip()]


def compute_path_length(G, path, pos):
    """Compute path length in meters using graph edge lengths."""
    total = 0.0
    for i in range(len(path) - 1):
        if G.has_edge(path[i], path[i + 1]):
            total += G[path[i]][path[i + 1]]["length"]
    return total


def get_path_edges(path):
    """Convert a node path to a list of edges."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def plot_comparison(G, pos, original_path, noisy_path, user_id, dataset_name, save=False, output_dir=None):
    """
    Plot original vs noisy path overlaid on the road network.
    """
    # IEEE single-column width ~3.5in, double-column ~7.16in
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))

    orig_edges = set(get_path_edges(original_path))
    noisy_edges = set(get_path_edges(noisy_path))
    common_edges = orig_edges & noisy_edges
    only_orig = orig_edges - noisy_edges
    only_noisy = noisy_edges - orig_edges

    # Compute stats
    orig_len = compute_path_length(G, original_path, pos)
    noisy_len = compute_path_length(G, noisy_path, pos)
    detour_pct = ((noisy_len - orig_len) / orig_len * 100) if orig_len > 0 else 0

    # Collect all nodes in both paths for bounding box
    all_path_nodes = set(original_path) | set(noisy_path)
    path_xs = [pos[n][0] for n in all_path_nodes if n in pos]
    path_ys = [pos[n][1] for n in all_path_nodes if n in pos]

    if not path_xs:
        print(f"Warning: No node positions found for user {user_id}")
        return

    # Add padding around the path area
    margin_x = (max(path_xs) - min(path_xs)) * 0.15 + 0.002
    margin_y = (max(path_ys) - min(path_ys)) * 0.15 + 0.002
    xlim = (min(path_xs) - margin_x, max(path_xs) + margin_x)
    ylim = (min(path_ys) - margin_y, max(path_ys) + margin_y)

    # Get network edges within the bounding box for background
    bg_edges = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if (xlim[0] <= x1 <= xlim[1] and ylim[0] <= y1 <= ylim[1]) or \
               (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                bg_edges.append((u, v))

    subplot_labels = ["(a)", "(b)", "(c)"]

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(subplot_labels[ax_idx])
        ax.tick_params(labelsize=6, direction="in", length=2)
        ax.tick_params(axis="x", rotation=30)

        # Draw background network (light gray)
        for u, v in bg_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#d5d5d5", linewidth=0.3, zorder=1)

        if ax_idx == 0:
            # Original path only
            for u, v in orig_edges:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#1565C0", linewidth=1.2, zorder=3)

        elif ax_idx == 1:
            # Noisy path only
            for u, v in noisy_edges:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#D84315", linewidth=1.2, zorder=3)

        elif ax_idx == 2:
            # Overlay: common edges in green, original-only in blue, noisy-only in red
            for u, v in common_edges:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#2E7D32", linewidth=1.2, zorder=3)

            for u, v in only_orig:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#1565C0", linewidth=1.2, zorder=4,
                            linestyle="--")

            for u, v in only_noisy:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#D84315", linewidth=1.2, zorder=4,
                            linestyle="--")

            # Legend for overlay
            legend_items = [
                mpatches.Patch(color="#2E7D32", label="Shared"),
                mpatches.Patch(color="#1565C0", label="Original only"),
                mpatches.Patch(color="#D84315", label="Detour"),
            ]
            ax.legend(handles=legend_items, loc="upper left", fontsize=7,
                      framealpha=0.9, edgecolor="0.7", handlelength=1.2)

        # Mark origin and destination
        orig_node = original_path[0]
        dest_node = original_path[-1]
        if orig_node in pos:
            ax.scatter(*pos[orig_node], color="#2E7D32", s=40, zorder=5,
                      edgecolors="black", linewidth=0.6, marker="o", label="Origin")
        if dest_node in pos:
            ax.scatter(*pos[dest_node], color="#C62828", s=40, zorder=5,
                      edgecolors="black", linewidth=0.6, marker="s", label="Dest.")

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
                      edgecolor="0.7", handlelength=1.2)

    plt.tight_layout(pad=0.4, w_pad=0.5)

    if save:
        out_dir = Path(output_dir) if output_dir else Path(__file__).parent
        out_path = out_dir / f"path_comparison_{user_id}_{dataset_name}.pdf"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved to {out_path}")
        # Also save PNG for quick preview
        out_png = out_dir / f"path_comparison_{user_id}_{dataset_name}.png"
        fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
        print(f"Saved to {out_png}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original vs noisy path for a user"
    )
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="User ID to visualize (e.g., p_81). If not set, picks first modified user."
    )
    parser.add_argument(
        "--dataset", type=str, default="ci", choices=["ci", "co", "cl"],
        help="Which dataset to use (default: ci)"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Pick a random user that has a different noisy path"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the plot to a PNG file"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing data files (default: ./data)"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"

    dataset_map = {
        "ci": ("subnet_ci.csv", "subnet_ci_noisy.csv"),
        "co": ("subnet_co.csv", "subnet_co_noisy.csv"),
        "cl": ("subnet_cl.csv", "subnet_cl_noisy.csv"),
    }

    orig_file, noisy_file = dataset_map[args.dataset]
    orig_path = data_dir / orig_file
    noisy_path = data_dir / noisy_file

    if not noisy_path.exists():
        print(f"Error: Noisy dataset not found at {noisy_path}")
        print("Run add_noise_to_paths.py first to generate noisy datasets.")
        return

    # Load data
    print("Loading road network...")
    G, pos = load_network(str(data_dir / "subnet_NY.json"))
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("Loading datasets...")
    df_orig = pd.read_csv(str(orig_path))
    df_noisy = pd.read_csv(str(noisy_path))

    # Select user
    if args.user_id:
        user_id = args.user_id
    else:
        # Find users whose path actually changed
        changed_users = []
        for idx, row in df_orig.iterrows():
            noisy_row = df_noisy.iloc[idx]
            if str(row["q_path"]) != str(noisy_row["q_path"]):
                changed_users.append(row["user_id"])

        if not changed_users:
            print("No modified paths found. Try different noise parameters.")
            return

        if args.random:
            user_id = np.random.choice(changed_users)
        else:
            user_id = changed_users[0]

    print(f"Visualizing user: {user_id}")

    # Get original and noisy paths
    orig_row = df_orig[df_orig["user_id"] == user_id]
    noisy_row = df_noisy[df_noisy["user_id"] == user_id]

    if orig_row.empty:
        print(f"Error: User {user_id} not found in {orig_file}")
        return
    if noisy_row.empty:
        print(f"Error: User {user_id} not found in {noisy_file}")
        return

    original_path_nodes = parse_path(str(orig_row.iloc[0]["q_path"]))
    noisy_path_nodes = parse_path(str(noisy_row.iloc[0]["q_path"]))

    if original_path_nodes == noisy_path_nodes:
        print(f"Note: Path for {user_id} was not modified (identical to original).")
        print("Try --random to pick a user with a modified path.")

    print(f"  Original path: {len(original_path_nodes)} nodes")
    print(f"  Noisy path:    {len(noisy_path_nodes)} nodes")

    plot_comparison(
        G, pos,
        original_path_nodes, noisy_path_nodes,
        user_id, args.dataset,
        save=args.save,
        output_dir=str(data_dir.parent),
    )


if __name__ == "__main__":
    main()
