"""
Spiking Neural Network utilities for spike encoding and decoding.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def encode_to_spikes(
    data: np.ndarray,
    method: str = 'rate',
    num_steps: int = 100,
    max_rate: float = 1.0,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Encode continuous data into spike trains.
    
    Args:
        data: Input data array [N, features]
        method: Encoding method ('rate', 'latency', 'burst')
        num_steps: Number of time steps for spike train
        max_rate: Maximum firing rate (for rate encoding)
        threshold: Optional threshold for spike generation
        
    Returns:
        spikes: Binary spike trains [N, num_steps, features]
    """
    N, features = data.shape
    
    if method == 'rate':
        # Rate encoding: spike probability proportional to input value
        # Normalize data to [0, max_rate]
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        normalized = normalized * max_rate
        
        # Generate spikes based on Bernoulli distribution
        spikes = np.zeros((N, num_steps, features))
        for t in range(num_steps):
            spikes[:, t, :] = (np.random.random((N, features)) < normalized).astype(float)
        
        return spikes
    
    elif method == 'latency':
        # Latency encoding: spike timing encodes value
        # Higher values spike earlier
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Convert to spike times (0 = earliest, num_steps-1 = latest)
        spike_times = ((1 - normalized) * (num_steps - 1)).astype(int)
        
        # Create spike trains
        spikes = np.zeros((N, num_steps, features))
        for i in range(N):
            for j in range(features):
                t = spike_times[i, j]
                if 0 <= t < num_steps:
                    spikes[i, t, j] = 1.0
        
        return spikes
    
    elif method == 'burst':
        # Burst encoding: number of spikes encodes value
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Number of spikes proportional to value
        num_spikes = (normalized * (num_steps // 2)).astype(int)
        
        spikes = np.zeros((N, num_steps, features))
        for i in range(N):
            for j in range(features):
                n = num_spikes[i, j]
                if n > 0:
                    # Random spike times
                    spike_times = np.random.choice(num_steps, size=min(n, num_steps), replace=False)
                    spikes[i, spike_times, j] = 1.0
        
        return spikes
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def decode_from_spikes(
    spikes: np.ndarray,
    method: str = 'rate',
    time_window: Optional[int] = None
) -> np.ndarray:
    """
    Decode spike trains back to continuous values.
    
    Args:
        spikes: Binary spike trains [N, num_steps, features]
        method: Decoding method ('rate', 'latency', 'weighted')
        time_window: Optional time window for rate calculation
        
    Returns:
        decoded: Continuous values [N, features]
    """
    N, num_steps, features = spikes.shape
    
    if method == 'rate':
        # Rate decoding: count spikes over time
        if time_window is None:
            time_window = num_steps
        
        decoded = spikes.sum(axis=1) / time_window
        return decoded
    
    elif method == 'latency':
        # Latency decoding: time of first spike
        decoded = np.zeros((N, features))
        
        for i in range(N):
            for j in range(features):
                spike_indices = np.where(spikes[i, :, j] > 0)[0]
                if len(spike_indices) > 0:
                    first_spike = spike_indices[0]
                    # Convert to normalized value (earlier = higher)
                    decoded[i, j] = 1.0 - (first_spike / num_steps)
                else:
                    decoded[i, j] = 0.0
        
        return decoded
    
    elif method == 'weighted':
        # Weighted decoding: recent spikes weighted more
        weights = np.linspace(0.5, 1.0, num_steps)
        weights = weights.reshape(1, num_steps, 1)
        
        weighted_spikes = spikes * weights
        decoded = weighted_spikes.sum(axis=1) / weights.sum()
        
        return decoded
    
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def calculate_spike_rate(
    spikes: np.ndarray,
    time_window: Optional[int] = None,
    dt: float = 1.0
) -> np.ndarray:
    """
    Calculate instantaneous firing rate from spike trains.
    
    Args:
        spikes: Binary spike trains [N, num_steps, features]
        time_window: Window size for rate calculation (None = all)
        dt: Time step duration (ms)
        
    Returns:
        rates: Firing rates in spikes/second [N, num_steps, features]
    """
    if time_window is None:
        # Average rate over entire sequence
        return spikes.mean(axis=1, keepdims=True).repeat(spikes.shape[1], axis=1)
    
    # Sliding window rate calculation
    N, num_steps, features = spikes.shape
    rates = np.zeros_like(spikes)
    
    for t in range(num_steps):
        start = max(0, t - time_window + 1)
        end = t + 1
        window_spikes = spikes[:, start:end, :]
        rates[:, t, :] = window_spikes.mean(axis=1) / (dt / 1000.0)  # Convert to Hz
    
    return rates


def poisson_spike_generator(
    rates: np.ndarray,
    dt: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Poisson spike trains from firing rates.
    
    Args:
        rates: Firing rates [N, features] in Hz
        dt: Time step duration in ms
        seed: Random seed for reproducibility
        
    Returns:
        spikes: Binary spike train [N, features]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert rate to probability per time step
    # rate (Hz) * dt (ms) / 1000 = probability
    prob = rates * dt / 1000.0
    prob = np.clip(prob, 0, 1)
    
    # Generate spikes
    spikes = (np.random.random(rates.shape) < prob).astype(float)
    
    return spikes


def encode_price_movements(
    prices: np.ndarray,
    returns_window: int = 1,
    num_steps: int = 100,
    method: str = 'rate'
) -> np.ndarray:
    """
    Encode price movements as spike trains.
    Positive returns -> high spike rate, negative -> low spike rate.
    
    Args:
        prices: Price time series [N, assets]
        returns_window: Window for return calculation
        num_steps: Number of spike train steps
        method: Encoding method
        
    Returns:
        spikes: Spike-encoded returns [N, num_steps, assets]
    """
    # Calculate returns
    if len(prices) <= returns_window:
        returns = np.zeros_like(prices)
    else:
        returns = np.zeros_like(prices)
        returns[returns_window:] = (
            prices[returns_window:] - prices[:-returns_window]
        ) / (prices[:-returns_window] + 1e-8)
    
    # Encode returns
    spikes = encode_to_spikes(returns, method=method, num_steps=num_steps)
    
    return spikes


def decode_trading_signals(
    spikes: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Decode spike trains into trading signals.
    
    Args:
        spikes: Spike trains [N, num_steps, 3] for [buy, hold, sell]
        threshold: Decision threshold
        
    Returns:
        signals: Trading signals [N] (-1=sell, 0=hold, 1=buy)
    """
    # Average spike rates per action
    rates = spikes.mean(axis=1)  # [N, 3]
    
    # Determine action with highest rate
    actions = rates.argmax(axis=1)  # [N]
    
    # Convert to [-1, 0, 1]
    # 0 -> 1 (buy), 1 -> 0 (hold), 2 -> -1 (sell)
    signals = np.where(actions == 0, 1, np.where(actions == 1, 0, -1))
    
    return signals


def spike_train_distance(
    spikes1: np.ndarray,
    spikes2: np.ndarray,
    metric: str = 'victor-purpura',
    cost: float = 0.5
) -> float:
    """
    Calculate distance between two spike trains.
    
    Args:
        spikes1: First spike train [num_steps, features]
        spikes2: Second spike train [num_steps, features]
        metric: Distance metric ('victor-purpura', 'van-rossum', 'cosine')
        cost: Cost parameter for spike timing differences
        
    Returns:
        distance: Spike train distance
    """
    if metric == 'cosine':
        # Simple cosine similarity
        flat1 = spikes1.flatten()
        flat2 = spikes2.flatten()
        
        dot = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = dot / (norm1 * norm2)
        return 1.0 - similarity
    
    elif metric == 'victor-purpura':
        # Simplified Victor-Purpura distance
        # Count insertions/deletions and shifts
        diff = np.abs(spikes1 - spikes2).sum()
        return diff
    
    elif metric == 'van-rossum':
        # Simplified van Rossum distance
        # Convolved spike trains with exponential kernel
        tau = 10.0  # Time constant
        num_steps = spikes1.shape[0]
        
        # Simple exponential kernel
        kernel = np.exp(-np.arange(num_steps) / tau)
        
        # Convolve
        conv1 = np.array([
            np.convolve(spikes1[:, i], kernel, mode='same')
            for i in range(spikes1.shape[1])
        ]).T
        
        conv2 = np.array([
            np.convolve(spikes2[:, i], kernel, mode='same')
            for i in range(spikes2.shape[1])
        ]).T
        
        # Calculate distance
        distance = np.sum((conv1 - conv2) ** 2)
        return distance
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_spike_raster_plot_data(
    spikes: np.ndarray,
    neuron_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare spike data for raster plot visualization.
    
    Args:
        spikes: Binary spike trains [num_steps, num_neurons]
        neuron_indices: Optional specific neurons to plot
        
    Returns:
        times: Spike times
        neurons: Neuron indices
    """
    if neuron_indices is None:
        neuron_indices = np.arange(spikes.shape[1])
    
    times = []
    neurons = []
    
    for i, neuron_idx in enumerate(neuron_indices):
        spike_times = np.where(spikes[:, neuron_idx] > 0)[0]
        times.extend(spike_times)
        neurons.extend([i] * len(spike_times))
    
    return np.array(times), np.array(neurons)
