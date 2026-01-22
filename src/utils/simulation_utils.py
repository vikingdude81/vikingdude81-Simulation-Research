"""
Utility functions for complex systems simulation
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import hashlib
import json


def generate_network_topology(n_nodes: int, 
                              topology_type: str = 'random',
                              connectivity: float = 0.1) -> List[Tuple[int, int]]:
    """
    Generate network topology for agent connections
    
    Args:
        n_nodes: Number of nodes in network
        topology_type: Type of topology ('random', 'ring', 'small_world', 'scale_free')
        connectivity: Connection probability for random networks
        
    Returns:
        List of (node_i, node_j) edge tuples
    """
    edges = []
    
    if topology_type == 'random':
        # Erdős-Rényi random graph
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < connectivity:
                    edges.append((i, j))
    
    elif topology_type == 'ring':
        # Ring network where each node connects to k nearest neighbors
        k = max(2, int(n_nodes * connectivity))
        for i in range(n_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n_nodes
                edges.append((i, neighbor))
    
    elif topology_type == 'small_world':
        # Watts-Strogatz small world network
        k = max(2, int(n_nodes * connectivity))
        rewire_prob = 0.1
        
        # Start with ring
        for i in range(n_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n_nodes
                # Rewire with probability
                if np.random.random() < rewire_prob:
                    new_neighbor = np.random.randint(n_nodes)
                    if new_neighbor != i:
                        edges.append((i, new_neighbor))
                else:
                    edges.append((i, neighbor))
    
    elif topology_type == 'scale_free':
        # Barabási-Albert scale-free network
        m = max(1, int(n_nodes * connectivity / 2))
        
        # Start with small complete graph
        for i in range(m):
            for j in range(i + 1, m):
                edges.append((i, j))
        
        # Add nodes with preferential attachment
        for i in range(m, n_nodes):
            # Calculate degree distribution
            degrees = [0] * i
            for edge in edges:
                if edge[0] < i:
                    degrees[edge[0]] += 1
                if edge[1] < i:
                    degrees[edge[1]] += 1
            
            # Preferential attachment
            total_degree = sum(degrees)
            if total_degree > 0 and i > 0:
                probs = np.array(degrees) / total_degree
                # Ensure we don't try to choose more nodes than available
                num_targets = min(m, i)
                if num_targets > 0:
                    targets = np.random.choice(i, size=num_targets, replace=False, p=probs)
                    for target in targets:
                        edges.append((i, target))
    
    return edges


def calculate_network_metrics(edges: List[Tuple[int, int]], 
                              n_nodes: int) -> Dict[str, float]:
    """
    Calculate basic network metrics
    
    Args:
        edges: List of network edges
        n_nodes: Total number of nodes
        
    Returns:
        Dictionary of network metrics
    """
    # Build adjacency list
    adjacency = {i: [] for i in range(n_nodes)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    # Calculate degree distribution
    degrees = [len(adjacency[i]) for i in range(n_nodes)]
    
    # Calculate clustering coefficient (simplified)
    clustering_coeffs = []
    for node in range(n_nodes):
        neighbors = adjacency[node]
        if len(neighbors) < 2:
            continue
        
        # Count triangles
        triangles = 0
        possible = len(neighbors) * (len(neighbors) - 1) // 2
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if n2 in adjacency[n1]:
                    triangles += 1
        
        if possible > 0:
            clustering_coeffs.append(triangles / possible)
    
    return {
        'avg_degree': np.mean(degrees),
        'degree_std': np.std(degrees),
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0,
        'avg_clustering': np.mean(clustering_coeffs) if clustering_coeffs else 0,
        'density': len(edges) / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
    }


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data array
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data array
    """
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    elif method == 'robust':
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(data)
        return (data - median) / iqr
    
    return data


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate Gini coefficient for inequality measurement
    
    Args:
        values: List of values (e.g., wealth distribution)
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if not values or all(v == 0 for v in values):
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    total = sum(sorted_values)
    
    # Guard against zero or negative total
    if total <= 0:
        return 0.0
    
    # Calculate Gini coefficient
    return (2 * sum((i + 1) * v for i, v in enumerate(sorted_values)) - 
            (n + 1) * total) / (n * total)


def generate_simulation_id(config: Dict[str, Any]) -> str:
    """
    Generate unique simulation ID based on configuration
    
    Args:
        config: Simulation configuration dictionary
        
    Returns:
        Unique simulation identifier
    """
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:16]


def calculate_entropy(distribution: List[float]) -> float:
    """
    Calculate Shannon entropy of a distribution
    
    Args:
        distribution: Probability distribution
        
    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    total = sum(distribution)
    if total == 0:
        return 0.0
    
    probs = [x / total for x in distribution if x > 0]
    return -sum(p * np.log2(p) for p in probs)


def moving_average(data: List[float], window: int = 5) -> List[float]:
    """
    Calculate moving average of data
    
    Args:
        data: Input data
        window: Window size for moving average
        
    Returns:
        Smoothed data
    """
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        end = i + 1
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def detect_regime_changes(time_series: List[float], 
                         threshold: float = 2.0) -> List[int]:
    """
    Detect regime changes in time series data
    
    Args:
        time_series: Input time series
        threshold: Threshold for detecting significant changes (in std devs)
        
    Returns:
        List of indices where regime changes occur
    """
    if len(time_series) < 3:
        return []
    
    # Calculate rolling statistics
    changes = np.diff(time_series)
    std = np.std(changes)
    
    if std == 0:
        return []
    
    # Detect significant changes
    regime_changes = []
    for i, change in enumerate(changes):
        if abs(change) > threshold * std:
            regime_changes.append(i + 1)
    
    return regime_changes


class ExperimentLogger:
    """Logger for simulation experiments"""
    
    def __init__(self, experiment_name: str):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.logs: List[Dict[str, Any]] = []
    
    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a simulation step
        
        Args:
            step: Simulation step number
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            'step': step,
            'timestamp': step,  # Can be replaced with actual timestamp
            **metrics
        }
        self.logs.append(log_entry)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get history of a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values over time
        """
        return [log.get(metric_name, 0) for log in self.logs]
    
    def save(self, filepath: str) -> None:
        """
        Save logs to file
        
        Args:
            filepath: Path to save logs
        """
        with open(filepath, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'logs': self.logs
            }, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("Testing network generation...")
    edges = generate_network_topology(20, 'small_world', 0.2)
    metrics = calculate_network_metrics(edges, 20)
    print(f"Generated network with {len(edges)} edges")
    print(f"Network metrics: {metrics}")
    
    print("\nTesting Gini coefficient...")
    wealth = [100, 200, 300, 400, 1000]
    gini = calculate_gini_coefficient(wealth)
    print(f"Gini coefficient: {gini:.3f}")
