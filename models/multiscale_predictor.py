"""
Multiscale Market Encoder
Based on arXiv:2512.12462 multiscale framework

Handles multiple timeframes (1H, 4H, 12H, 1D, 1W) with different sampling rates.
Implements nonlinear aggregation across timeframes and real-time recursive decoding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiscaleMarketEncoder(nn.Module):
    """
    Encoder handling multiple timeframes with different sampling rates.
    Based on arXiv:2512.12462 multiscale framework.
    
    Features:
    - Encode 1H, 4H, 12H, 1D, 1W simultaneously
    - Handle missing data (exchange outages)
    - Nonlinear aggregation across timeframes
    - Real-time recursive decoding
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_scales: int = 5,  # 1H, 4H, 12H, 1D, 1W
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize MultiscaleMarketEncoder.
        
        Args:
            input_dim: Number of input features per timeframe
            hidden_dim: Hidden dimension for processing each scale
            num_scales: Number of timeframes to process
            output_dim: Output dimension of encoded representation
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.output_dim = output_dim
        
        # Per-scale encoders (separate for each timeframe)
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_scales)
        ])
        
        # Missing data handler - learns to impute or downweight
        self.missing_handler = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),  # Input: missing indicator
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])
        
        # Nonlinear cross-scale aggregation
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Recursive decoder for real-time updates
        self.recursive_decoder = nn.GRUCell(output_dim, output_dim)
        
    def forward(
        self,
        multi_scale_data: Dict[str, torch.Tensor],
        missing_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through multiscale encoder.
        
        Args:
            multi_scale_data: Dict with keys '1H', '4H', '12H', '1D', '1W'
                             Each tensor has shape [batch_size, input_dim]
            missing_masks: Optional dict with same keys, tensors of shape [batch_size, 1]
                          1 = data present, 0 = data missing
                          
        Returns:
            encoded: Aggregated multiscale representation [batch_size, output_dim]
            scale_features: Dict of per-scale encoded features
        """
        scale_keys = ['1H', '4H', '12H', '1D', '1W']
        batch_size = multi_scale_data[scale_keys[0]].shape[0]
        device = multi_scale_data[scale_keys[0]].device
        
        # Create default missing masks if not provided
        if missing_masks is None:
            missing_masks = {
                key: torch.ones(batch_size, 1, device=device)
                for key in scale_keys
            }
        
        # Encode each scale
        scale_features = []
        scale_features_dict = {}
        
        for i, scale_key in enumerate(scale_keys):
            # Get data and missing mask
            scale_data = multi_scale_data[scale_key]
            missing_mask = missing_masks.get(scale_key, torch.ones(batch_size, 1, device=device))
            
            # Encode scale
            encoded = self.scale_encoders[i](scale_data)
            
            # Handle missing data
            missing_weight = self.missing_handler[i](missing_mask)
            # Broadcast weight to match encoded dimensions
            missing_weight = missing_weight.expand(-1, self.hidden_dim)
            
            # Apply missing data weighting
            weighted_encoded = encoded * missing_weight
            
            scale_features.append(weighted_encoded)
            scale_features_dict[scale_key] = weighted_encoded
        
        # Stack for attention: [batch_size, num_scales, hidden_dim]
        scale_stack = torch.stack(scale_features, dim=1)
        
        # Cross-scale attention
        attended_features, _ = self.cross_scale_attention(
            scale_stack, scale_stack, scale_stack
        )
        
        # Flatten for aggregation: [batch_size, num_scales * hidden_dim]
        flattened = attended_features.reshape(batch_size, -1)
        
        # Final aggregation
        encoded = self.aggregation(flattened)
        
        return encoded, scale_features_dict
    
    def recursive_decode(
        self,
        encoded: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recursive decoding for real-time updates.
        
        Args:
            encoded: Current encoded state [batch_size, output_dim]
            hidden_state: Previous hidden state [batch_size, output_dim]
                         If None, initializes to zeros
                         
        Returns:
            output: Decoded output [batch_size, output_dim]
            new_hidden: Updated hidden state [batch_size, output_dim]
        """
        if hidden_state is None:
            hidden_state = torch.zeros_like(encoded)
        
        new_hidden = self.recursive_decoder(encoded, hidden_state)
        return new_hidden, new_hidden
    
    def encode_timeframe_sequence(
        self,
        multi_scale_sequences: Dict[str, torch.Tensor],
        missing_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Encode sequences across multiple timeframes.
        
        Args:
            multi_scale_sequences: Dict with keys '1H', '4H', '12H', '1D', '1W'
                                  Each tensor has shape [batch_size, seq_len, input_dim]
            missing_masks: Optional dict with same keys, 
                          tensors of shape [batch_size, seq_len, 1]
                          
        Returns:
            sequence_encodings: Encoded sequences [batch_size, seq_len, output_dim]
        """
        scale_keys = ['1H', '4H', '12H', '1D', '1W']
        batch_size, seq_len, _ = multi_scale_sequences[scale_keys[0]].shape
        
        # Process each timestep
        encoded_sequence = []
        hidden_state = None
        
        for t in range(seq_len):
            # Extract data at timestep t
            timestep_data = {
                key: multi_scale_sequences[key][:, t, :]
                for key in scale_keys
            }
            
            # Extract missing masks at timestep t if provided
            timestep_masks = None
            if missing_masks is not None:
                timestep_masks = {
                    key: missing_masks[key][:, t, :]
                    for key in scale_keys
                }
            
            # Encode timestep
            encoded, _ = self.forward(timestep_data, timestep_masks)
            
            # Recursive decode
            decoded, hidden_state = self.recursive_decode(encoded, hidden_state)
            encoded_sequence.append(decoded)
        
        # Stack into sequence: [batch_size, seq_len, output_dim]
        return torch.stack(encoded_sequence, dim=1)


class MultiscalePredictor(nn.Module):
    """
    Complete multiscale predictor with encoder and prediction head.
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_scales: int = 5,
        output_dim: int = 64,
        num_predictions: int = 1,  # Number of steps to predict
        dropout: float = 0.1
    ):
        """
        Initialize MultiscalePredictor.
        
        Args:
            input_dim: Number of input features per timeframe
            hidden_dim: Hidden dimension for encoder
            num_scales: Number of timeframes
            output_dim: Encoder output dimension
            num_predictions: Number of future steps to predict
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = MultiscaleMarketEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predictions)
        )
        
    def forward(
        self,
        multi_scale_data: Dict[str, torch.Tensor],
        missing_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with prediction.
        
        Args:
            multi_scale_data: Multi-timeframe input data
            missing_masks: Optional missing data masks
            
        Returns:
            predictions: Predicted values [batch_size, num_predictions]
            encoded: Encoded representation [batch_size, output_dim]
            scale_features: Per-scale features
        """
        encoded, scale_features = self.encoder(multi_scale_data, missing_masks)
        predictions = self.prediction_head(encoded)
        return predictions, encoded, scale_features
