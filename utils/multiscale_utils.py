"""
Multiscale utilities for timeframe aggregation and handling.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional


def aggregate_timeframes(
    data_dict: Dict[str, pd.DataFrame],
    timeframes: List[str] = ['1H', '4H', '12H', '1D', '1W']
) -> Dict[str, np.ndarray]:
    """
    Aggregate data from multiple timeframes into aligned arrays.
    
    Args:
        data_dict: Dictionary mapping timeframe to DataFrame
        timeframes: List of timeframe keys to aggregate
        
    Returns:
        aggregated: Dictionary mapping timeframe to numpy arrays
    """
    aggregated = {}
    
    for tf in timeframes:
        if tf in data_dict:
            df = data_dict[tf]
            # Convert to numpy, handling any non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            aggregated[tf] = df[numeric_cols].values
        else:
            # Return empty array if timeframe not available
            aggregated[tf] = np.array([])
    
    return aggregated


def create_missing_mask(
    data_dict: Dict[str, np.ndarray],
    timeframes: List[str] = ['1H', '4H', '12H', '1D', '1W']
) -> Dict[str, np.ndarray]:
    """
    Create binary masks indicating missing data.
    
    Args:
        data_dict: Dictionary mapping timeframe to data arrays
        timeframes: List of timeframe keys
        
    Returns:
        masks: Dictionary mapping timeframe to binary masks (1=present, 0=missing)
    """
    masks = {}
    
    for tf in timeframes:
        if tf in data_dict and len(data_dict[tf]) > 0:
            data = data_dict[tf]
            # Check for NaN or infinite values
            mask = ~(np.isnan(data) | np.isinf(data))
            # Create row-wise mask (all features present)
            row_mask = mask.all(axis=1).astype(float)
            masks[tf] = row_mask.reshape(-1, 1)
        else:
            masks[tf] = np.array([[0.0]])  # All missing
    
    return masks


def interpolate_missing_data(
    data: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate missing data in array.
    
    Args:
        data: Input array with potential NaN values
        method: Interpolation method ('linear', 'forward', 'backward')
        
    Returns:
        interpolated: Array with missing values filled
    """
    df = pd.DataFrame(data)
    
    if method == 'linear':
        interpolated = df.interpolate(method='linear', limit_direction='both')
    elif method == 'forward':
        interpolated = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'backward':
        interpolated = df.fillna(method='bfill').fillna(method='ffill')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Fill any remaining NaNs with 0
    interpolated = interpolated.fillna(0)
    
    return interpolated.values


def align_timeframes(
    data_dict: Dict[str, pd.DataFrame],
    reference_timeframe: str = '1H',
    target_length: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Align different timeframes to same length by resampling.
    
    Args:
        data_dict: Dictionary mapping timeframe to DataFrame
        reference_timeframe: Timeframe to use as length reference
        target_length: Optional target length (overrides reference)
        
    Returns:
        aligned: Dictionary with aligned DataFrames
    """
    if target_length is None:
        if reference_timeframe in data_dict:
            target_length = len(data_dict[reference_timeframe])
        else:
            # Use maximum length
            target_length = max(len(df) for df in data_dict.values())
    
    aligned = {}
    
    for tf, df in data_dict.items():
        if len(df) == target_length:
            aligned[tf] = df
        elif len(df) > target_length:
            # Downsample
            indices = np.linspace(0, len(df) - 1, target_length, dtype=int)
            aligned[tf] = df.iloc[indices].reset_index(drop=True)
        else:
            # Upsample using interpolation
            old_indices = np.arange(len(df))
            new_indices = np.linspace(0, len(df) - 1, target_length)
            
            # Interpolate each column
            new_data = {}
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    new_data[col] = np.interp(new_indices, old_indices, df[col].values)
                else:
                    # For non-numeric, just repeat
                    new_data[col] = df[col].iloc[0]
            
            aligned[tf] = pd.DataFrame(new_data)
    
    return aligned


def resample_to_timeframe(
    df: pd.DataFrame,
    target_timeframe: str,
    datetime_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Resample DataFrame to target timeframe.
    
    Args:
        df: Input DataFrame with datetime index or column
        target_timeframe: Target timeframe ('1H', '4H', '1D', etc.)
        datetime_col: Name of datetime column if not using index
        
    Returns:
        resampled: DataFrame resampled to target timeframe
    """
    # Convert timeframe string to pandas offset
    timeframe_map = {
        '1H': '1H',
        '4H': '4H',
        '12H': '12H',
        '1D': '1D',
        '1W': '1W'
    }
    
    if target_timeframe not in timeframe_map:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    offset = timeframe_map[target_timeframe]
    
    # Ensure datetime index
    if datetime_col in df.columns:
        df = df.set_index(datetime_col)
    
    # Resample
    resampled = df.resample(offset).agg({
        col: 'last' if col in ['close', 'open', 'high', 'low', 'volume'] else 'mean'
        for col in df.columns
    })
    
    return resampled.reset_index()


def create_multiscale_batch(
    data_dict: Dict[str, np.ndarray],
    timeframes: List[str] = ['1H', '4H', '12H', '1D', '1W'],
    device: str = 'cpu'
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create batch of multiscale data as PyTorch tensors.
    
    Args:
        data_dict: Dictionary mapping timeframe to data arrays
        timeframes: List of timeframe keys
        device: Device to place tensors on
        
    Returns:
        data_tensors: Dictionary of data tensors
        mask_tensors: Dictionary of mask tensors
    """
    # Create masks
    masks = create_missing_mask(data_dict, timeframes)
    
    data_tensors = {}
    mask_tensors = {}
    
    for tf in timeframes:
        if tf in data_dict and len(data_dict[tf]) > 0:
            # Interpolate missing values
            clean_data = interpolate_missing_data(data_dict[tf])
            data_tensors[tf] = torch.FloatTensor(clean_data).to(device)
            mask_tensors[tf] = torch.FloatTensor(masks[tf]).to(device)
        else:
            # Create dummy tensors
            data_tensors[tf] = torch.zeros(1, 10).to(device)  # Dummy shape
            mask_tensors[tf] = torch.zeros(1, 1).to(device)
    
    return data_tensors, mask_tensors


def calculate_timeframe_weights(
    timeframes: List[str] = ['1H', '4H', '12H', '1D', '1W']
) -> Dict[str, float]:
    """
    Calculate importance weights for different timeframes.
    Shorter timeframes get lower weights, longer get higher.
    
    Args:
        timeframes: List of timeframe keys
        
    Returns:
        weights: Dictionary mapping timeframe to weight
    """
    # Map timeframes to hours
    hours_map = {
        '1H': 1,
        '4H': 4,
        '12H': 12,
        '1D': 24,
        '1W': 168
    }
    
    hours = [hours_map.get(tf, 1) for tf in timeframes]
    total_hours = sum(hours)
    
    weights = {
        tf: h / total_hours
        for tf, h in zip(timeframes, hours)
    }
    
    return weights


def simulate_missing_data(
    data_dict: Dict[str, np.ndarray],
    missing_rate: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Simulate missing data by randomly masking entries.
    
    Args:
        data_dict: Dictionary of data arrays
        missing_rate: Fraction of data to mark as missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        corrupted_data: Data with NaN values inserted
        true_masks: True missing data masks
    """
    if seed is not None:
        np.random.seed(seed)
    
    corrupted_data = {}
    true_masks = {}
    
    for tf, data in data_dict.items():
        # Create copy
        corrupted = data.copy()
        
        # Generate random mask
        mask = np.random.random(data.shape) > missing_rate
        true_masks[tf] = mask.astype(float)
        
        # Insert NaN where mask is False
        corrupted[~mask] = np.nan
        corrupted_data[tf] = corrupted
    
    return corrupted_data, true_masks
