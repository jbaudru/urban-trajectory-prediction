"""
Visualization module for comparing predicted vs actual paths on a map.
"""

import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
import os
import argparse
from datetime import datetime

# Import models
from models import LSTMTrajectoryPredictor, TransformerTrajectoryPredictor, GNNTrajectoryPredictor


class PathVisualizer:
    """Visualize predicted vs actual paths on a map"""
    
    def __init__(self, data_path, graph_path, model_dir="model_outputs", sequence_length=10):
        self.data_path = data_path
        self.graph_path = graph_path
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data and graph
        self.load_data()
        self.load_graph()
        self.prepare_encoder()
        self.prepare_edge_index()
        
    def load_data(self):
        """Load trajectory data"""
        print("Loading trajectory data...")
        self.df = pd.read_csv(self.data_path)
        
        # Handle different column formats
        if 'q_path' in self.df.columns:
            def parse_q_path(path_str):
                if isinstance(path_str, str):
                    return [int(node.strip()) for node in path_str.split(',') if node.strip()]
                return []
            self.df['route_nodes'] = self.df['q_path'].apply(parse_q_path)
        elif 'route_taken' in self.df.columns:
            import ast
            self.df['route_nodes'] = self.df['route_taken'].apply(ast.literal_eval)
        
        # Filter short routes
        self.df = self.df[self.df['route_nodes'].apply(len) >= self.sequence_length + 2]
        print(f"Loaded {len(self.df)} trips")
        
    def load_graph(self):
        """Load network graph with node positions"""
        print("Loading network graph...")
        with open(self.graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Extract node positions
        self.node_positions = {}
        if isinstance(graph_data['nodes'], list):
            for node in graph_data['nodes']:
                node_id = node['id']
                if 'x' in node and 'y' in node:
                    self.node_positions[node_id] = (node['x'], node['y'])
        else:
            for node_id, node_data in graph_data['nodes'].items():
                if 'x' in node_data and 'y' in node_data:
                    self.node_positions[int(node_id)] = (node_data['x'], node_data['y'])
        
        # Extract edges for drawing the road network
        self.edges = []
        edges_key = 'links' if 'links' in graph_data else 'edges'
        if edges_key in graph_data:
            for edge in graph_data[edges_key]:
                source = edge.get('source')
                target = edge.get('target')
                if source in self.node_positions and target in self.node_positions:
                    self.edges.append((source, target))
        
        print(f"Loaded {len(self.node_positions)} node positions and {len(self.edges)} edges")
        
    def prepare_encoder(self):
        """Prepare node encoder from trajectory data"""
        all_nodes = set()
        for route in self.df['route_nodes']:
            all_nodes.update(route)
        
        self.node_encoder = LabelEncoder()
        self.node_encoder.fit(list(all_nodes))
        self.vocab_size = len(self.node_encoder.classes_)
        print(f"Vocabulary size: {self.vocab_size}")
        
    def prepare_edge_index(self):
        """Prepare edge index for GNN"""
        original_to_encoded = {node: i for i, node in enumerate(self.node_encoder.classes_)}
        
        edge_list = []
        for source, target in self.edges:
            src_enc = original_to_encoded.get(source)
            tgt_enc = original_to_encoded.get(target)
            if src_enc is not None and tgt_enc is not None:
                edge_list.append([src_enc, tgt_enc])
        
        if edge_list:
            self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.tensor([[i, i] for i in range(self.vocab_size)], dtype=torch.long).t().contiguous()
        
        # Add self-loops
        self_loops = torch.tensor([[i, i] for i in range(self.vocab_size)], dtype=torch.long).t().contiguous()
        self.edge_index = torch.cat([self.edge_index, self_loops], dim=1)
        self.edge_index = torch.unique(self.edge_index, dim=1)
        
    def load_model(self, model_type):
        """Load a trained model"""
        # Try K-Fold model first, then regular model
        kfold_path = os.path.join(self.model_dir, f'best_{model_type}_model_kfold.pth')
        regular_path = os.path.join(self.model_dir, f'best_{model_type}_model.pth')
        
        if os.path.exists(kfold_path):
            model_path = kfold_path
        elif os.path.exists(regular_path):
            model_path = regular_path
        else:
            print(f"No trained model found for {model_type}")
            return None
        
        print(f"Loading {model_type} model from {model_path}")
        
        if model_type == 'lstm':
            model = LSTMTrajectoryPredictor(self.vocab_size)
        elif model_type == 'transformer':
            model = TransformerTrajectoryPredictor(self.vocab_size)
        elif model_type == 'gnn':
            model = GNNTrajectoryPredictor(self.vocab_size)
        else:
            return None
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_path(self, model, initial_sequence, num_steps, model_type):
        """Predict full path autoregressively"""
        model.eval()
        
        current_sequence = initial_sequence.unsqueeze(0).to(self.device)
        predicted_path = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                if model_type == 'gnn':
                    edge_index = self.edge_index.to(self.device)
                    output = model(current_sequence, edge_index)
                else:
                    output = model(current_sequence)
                
                next_node = torch.argmax(output, dim=1)
                predicted_path.append(next_node.item())
                
                current_sequence = torch.cat([
                    current_sequence[:, 1:],
                    next_node.unsqueeze(1)
                ], dim=1)
        
        return predicted_path
    
    def get_path_coordinates(self, encoded_path):
        """Convert encoded path to coordinates"""
        decoded_path = self.node_encoder.inverse_transform(encoded_path)
        coords = []
        for node in decoded_path:
            if node in self.node_positions:
                coords.append(self.node_positions[node])
            else:
                coords.append(None)
        return coords, decoded_path
    
    def visualize_single_path(self, ax, actual_path, predicted_path, title="", show_network=True):
        """Visualize a single actual vs predicted path comparison"""
        # Get coordinates
        actual_coords, actual_nodes = self.get_path_coordinates(actual_path)
        pred_coords, pred_nodes = self.get_path_coordinates(predicted_path)
        
        # Filter out None values
        actual_coords = [(x, y) for x, y in actual_coords if x is not None]
        pred_coords = [(x, y) for x, y in pred_coords if x is not None]
        
        if not actual_coords or not pred_coords:
            return False
        
        # Draw light road network background
        if show_network:
            for source, target in self.edges[:5000]:  # Limit for performance
                if source in self.node_positions and target in self.node_positions:
                    x1, y1 = self.node_positions[source]
                    x2, y2 = self.node_positions[target]
                    ax.plot([x1, x2], [y1, y2], 'lightgray', linewidth=0.3, alpha=0.5)
        
        # Plot actual path (green)
        actual_x = [c[0] for c in actual_coords]
        actual_y = [c[1] for c in actual_coords]
        ax.plot(actual_x, actual_y, 'g-', linewidth=2.5, label='Actual Path', alpha=0.8)
        ax.scatter(actual_x, actual_y, c='green', s=20, zorder=5)
        
        # Plot predicted path (red dashed)
        pred_x = [c[0] for c in pred_coords]
        pred_y = [c[1] for c in pred_coords]
        ax.plot(pred_x, pred_y, 'r--', linewidth=2.5, label='Predicted Path', alpha=0.8)
        ax.scatter(pred_x, pred_y, c='red', s=20, zorder=5, marker='x')
        
        # Mark start and end points
        ax.scatter([actual_x[0]], [actual_y[0]], c='blue', s=150, marker='o', zorder=10, label='Start')
        ax.scatter([actual_x[-1]], [actual_y[-1]], c='purple', s=150, marker='*', zorder=10, label='End (Actual)')
        
        # Calculate accuracy
        correct = sum(1 for a, p in zip(actual_path, predicted_path) if a == p)
        accuracy = correct / len(actual_path) * 100
        
        ax.set_title(f'{title}\nAccuracy: {accuracy:.1f}% ({correct}/{len(actual_path)} nodes)', fontsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        
        return True
    
    def visualize_comparisons(self, model_type='lstm', num_examples=10, save_path=None):
        """Generate visualization of multiple path comparisons"""
        print(f"\nGenerating path visualizations for {model_type.upper()} model...")
        
        # Load model
        model = self.load_model(model_type)
        if model is None:
            print(f"Could not load {model_type} model. Please train it first.")
            return
        
        # Get test routes
        test_routes = []
        for idx, route in enumerate(self.df['route_nodes'].values):
            if len(route) >= self.sequence_length + 5:  # Need some nodes to predict
                route_encoded = self.node_encoder.transform(route)
                test_routes.append((idx, route_encoded))
        
        # Randomly select examples
        np.random.seed(42)
        if len(test_routes) > num_examples:
            selected_indices = np.random.choice(len(test_routes), num_examples, replace=False)
            test_routes = [test_routes[i] for i in selected_indices]
        
        print(f"Selected {len(test_routes)} routes for visualization")
        
        # Create figure
        num_cols = 2
        num_rows = (num_examples + 1) // 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5 * num_rows))
        axes = axes.flatten() if num_examples > 1 else [axes]
        
        successful_plots = 0
        
        for i, (route_idx, route_encoded) in enumerate(test_routes):
            if i >= len(axes):
                break
                
            # Use first seq_len nodes as initial sequence
            initial_seq = torch.tensor(route_encoded[:self.sequence_length], dtype=torch.long)
            
            # Ground truth: remaining nodes
            actual_path = list(route_encoded[self.sequence_length:])
            num_steps = min(len(actual_path), 30)  # Limit prediction length
            actual_path = actual_path[:num_steps]
            
            if num_steps < 3:
                continue
            
            # Predict
            predicted_path = self.predict_path(model, initial_seq, num_steps, model_type)
            
            # Visualize
            success = self.visualize_single_path(
                axes[i], 
                actual_path, 
                predicted_path, 
                title=f"Route {route_idx + 1}",
                show_network=False  # Disable for cleaner visualization
            )
            
            if success:
                successful_plots += 1
        
        # Hide unused axes
        for i in range(successful_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Model: Predicted vs Actual Paths\n({successful_plots} examples)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = os.path.join(self.model_dir, f'{model_type}_path_comparison.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
        
        return save_path
    
    def visualize_all_models(self, num_examples=10):
        """Generate visualizations for all trained models"""
        model_types = ['lstm', 'transformer', 'gnn']
        saved_paths = []
        
        for model_type in model_types:
            try:
                path = self.visualize_comparisons(model_type, num_examples)
                if path:
                    saved_paths.append(path)
            except Exception as e:
                print(f"Error visualizing {model_type}: {e}")
        
        return saved_paths
    
    def create_interactive_map(self, model_type='lstm', num_examples=5, save_path=None):
        """Create an interactive HTML map using folium (if available)"""
        try:
            import folium
            from folium import plugins
        except ImportError:
            print("Folium not installed. Run: pip install folium")
            return None
        
        print(f"\nCreating interactive map for {model_type.upper()} model...")
        
        # Load model
        model = self.load_model(model_type)
        if model is None:
            return None
        
        # Get center coordinates
        all_lats = [pos[1] for pos in self.node_positions.values()]
        all_lons = [pos[0] for pos in self.node_positions.values()]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
        
        # Get test routes
        test_routes = []
        for idx, route in enumerate(self.df['route_nodes'].values):
            if len(route) >= self.sequence_length + 5:
                route_encoded = self.node_encoder.transform(route)
                test_routes.append((idx, route_encoded))
        
        np.random.seed(42)
        if len(test_routes) > num_examples:
            selected_indices = np.random.choice(len(test_routes), num_examples, replace=False)
            test_routes = [test_routes[i] for i in selected_indices]
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink']
        
        for i, (route_idx, route_encoded) in enumerate(test_routes):
            initial_seq = torch.tensor(route_encoded[:self.sequence_length], dtype=torch.long)
            actual_path = list(route_encoded[self.sequence_length:])
            num_steps = min(len(actual_path), 30)
            actual_path = actual_path[:num_steps]
            
            if num_steps < 3:
                continue
            
            predicted_path = self.predict_path(model, initial_seq, num_steps, model_type)
            
            # Get coordinates
            actual_coords, _ = self.get_path_coordinates(actual_path)
            pred_coords, _ = self.get_path_coordinates(predicted_path)
            
            actual_coords = [(y, x) for x, y in actual_coords if x is not None]  # lat, lon format
            pred_coords = [(y, x) for x, y in pred_coords if x is not None]
            
            if not actual_coords or not pred_coords:
                continue
            
            color = colors[i % len(colors)]
            
            # Draw actual path
            folium.PolyLine(
                actual_coords,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Route {i+1} - Actual"
            ).add_to(m)
            
            # Draw predicted path (dashed effect with markers)
            folium.PolyLine(
                pred_coords,
                color=color,
                weight=3,
                opacity=0.5,
                dash_array='10',
                popup=f"Route {i+1} - Predicted"
            ).add_to(m)
            
            # Add markers for start/end
            folium.Marker(
                actual_coords[0],
                popup=f"Route {i+1} Start",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            folium.Marker(
                actual_coords[-1],
                popup=f"Route {i+1} End",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                    background-color: white; padding: 10px; border-radius: 5px;
                    border: 2px solid gray;">
            <b>Legend</b><br>
            <span style="color: gray;">━━━ Actual Path</span><br>
            <span style="color: gray;">┅┅┅ Predicted Path</span><br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.model_dir, f'{model_type}_interactive_map.html')
        
        m.save(save_path)
        print(f"Saved interactive map to {save_path}")
        
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize predicted vs actual paths')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'gnn', 'all'],
                       default='all', help='Model to visualize')
    parser.add_argument('--data-path', type=str, default="data/subnet_ci.csv",
                       help='Path to data CSV')
    parser.add_argument('--graph-path', type=str, default="data/subnet_NY.json",
                       help='Path to graph JSON')
    parser.add_argument('--model-dir', type=str, default="model_outputs",
                       help='Directory with trained models')
    parser.add_argument('--num-examples', type=int, default=10,
                       help='Number of examples to visualize')
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive HTML map')
    
    args = parser.parse_args()
    
    visualizer = PathVisualizer(
        data_path=args.data_path,
        graph_path=args.graph_path,
        model_dir=args.model_dir
    )
    
    if args.model == 'all':
        visualizer.visualize_all_models(args.num_examples)
        if args.interactive:
            for model_type in ['lstm', 'transformer', 'gnn']:
                visualizer.create_interactive_map(model_type, min(args.num_examples, 5))
    else:
        visualizer.visualize_comparisons(args.model, args.num_examples)
        if args.interactive:
            visualizer.create_interactive_map(args.model, min(args.num_examples, 5))


if __name__ == "__main__":
    main()
