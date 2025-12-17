"""
Experiment 1: Timeframe Encoder Testing
Test MultiscaleMarketEncoder on BTC/ETH/SOL across all timeframes
Compare to current approach (separate models)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

from models.multiscale_predictor import MultiscaleMarketEncoder, MultiscalePredictor
from utils.multiscale_utils import (
    create_multiscale_batch,
    simulate_missing_data,
    interpolate_missing_data
)


def load_mock_data(asset: str = 'BTC', num_samples: int = 1000):
    """
    Load or generate mock multi-timeframe data for testing.
    
    In production, replace this with actual data loading from your data sources.
    """
    print(f"Loading data for {asset}...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    # Create data for each timeframe
    timeframes = {
        '1H': num_samples,
        '4H': num_samples // 4,
        '12H': num_samples // 12,
        '1D': num_samples // 24,
        '1W': num_samples // 168
    }
    
    data_dict = {}
    feature_dim = 50  # Number of features
    
    for tf, samples in timeframes.items():
        # Generate synthetic price-like features
        data = np.random.randn(samples, feature_dim) * 10 + 100
        # Add some trend
        trend = np.linspace(0, 20, samples).reshape(-1, 1)
        data = data + trend
        data_dict[tf] = data
    
    return data_dict


def create_baseline_model(input_dim: int, hidden_dim: int = 64):
    """Create baseline single-timeframe LSTM model."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )


def train_multiscale_model(
    model: MultiscalePredictor,
    data_dict: dict,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train multiscale predictor model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"\nTraining on device: {device}")
    
    # Prepare data
    timeframes = ['1H', '4H', '12H', '1D', '1W']
    
    # Use smallest timeframe length as reference
    min_length = min(data_dict[tf].shape[0] for tf in timeframes)
    
    # Create training batches
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, min_length - batch_size, batch_size):
            # Extract batch from each timeframe
            batch_data = {}
            for tf in timeframes:
                # Sample corresponding indices
                if tf == '1H':
                    batch_data[tf] = data_dict[tf][i:i+batch_size]
                else:
                    # Sample proportionally for longer timeframes
                    ratio = data_dict[tf].shape[0] / data_dict['1H'].shape[0]
                    start_idx = int(i * ratio)
                    end_idx = int((i + batch_size) * ratio)
                    if end_idx > data_dict[tf].shape[0]:
                        end_idx = data_dict[tf].shape[0]
                    batch_data[tf] = data_dict[tf][start_idx:end_idx]
                    
                    # Pad if needed
                    if batch_data[tf].shape[0] < batch_size:
                        pad_size = batch_size - batch_data[tf].shape[0]
                        padding = np.repeat(batch_data[tf][-1:], pad_size, axis=0)
                        batch_data[tf] = np.vstack([batch_data[tf], padding])
            
            # Convert to tensors
            batch_tensors = {}
            for tf in timeframes:
                batch_tensors[tf] = torch.FloatTensor(batch_data[tf]).to(device)
            
            # Create synthetic targets (e.g., next step prediction)
            targets = torch.FloatTensor(
                data_dict['1H'][i+1:i+batch_size+1, 0]
            ).unsqueeze(-1).to(device)
            
            if targets.shape[0] != batch_size:
                continue
            
            # Forward pass
            predictions, encoded, scale_features = model(batch_tensors)
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses


def evaluate_models(
    multiscale_model: MultiscalePredictor,
    baseline_model: nn.Module,
    test_data: dict,
    device: torch.device
):
    """Evaluate and compare models."""
    multiscale_model.eval()
    baseline_model.eval()
    
    timeframes = ['1H', '4H', '12H', '1D', '1W']
    
    with torch.no_grad():
        # Prepare test batch
        test_size = min(test_data[tf].shape[0] for tf in timeframes)
        batch_size = min(64, test_size)
        
        batch_data = {}
        for tf in timeframes:
            batch_data[tf] = torch.FloatTensor(
                test_data[tf][:batch_size]
            ).to(device)
        
        # Multiscale predictions
        predictions, _, _ = multiscale_model(batch_data)
        
        # Baseline predictions (using 1H data only)
        baseline_pred = baseline_model(batch_data['1H'])
        
        # Synthetic ground truth
        targets = torch.FloatTensor(
            test_data['1H'][1:batch_size+1, 0]
        ).unsqueeze(-1).to(device)
        
        if targets.shape[0] == predictions.shape[0]:
            multiscale_mse = nn.MSELoss()(predictions, targets).item()
            baseline_mse = nn.MSELoss()(baseline_pred[:len(targets)], targets).item()
            
            improvement = ((baseline_mse - multiscale_mse) / baseline_mse) * 100
            
            return {
                'multiscale_mse': multiscale_mse,
                'baseline_mse': baseline_mse,
                'improvement_pct': improvement
            }
    
    return None


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 1: Timeframe Encoder Testing")
    print("=" * 80)
    
    # Configuration
    assets = ['BTC', 'ETH', 'SOL']
    input_dim = 50
    hidden_dim = 128
    output_dim = 64
    
    results = {}
    
    for asset in assets:
        print(f"\n{'='*80}")
        print(f"Testing on {asset}")
        print(f"{'='*80}")
        
        # Load data
        data_dict = load_mock_data(asset, num_samples=1000)
        
        # Split train/test (80/20)
        train_data = {}
        test_data = {}
        for tf, data in data_dict.items():
            split_idx = int(0.8 * len(data))
            train_data[tf] = data[:split_idx]
            test_data[tf] = data[split_idx:]
        
        # Create models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        multiscale_model = MultiscalePredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_scales=5,
            output_dim=output_dim,
            num_predictions=1
        )
        
        baseline_model = create_baseline_model(input_dim, hidden_dim=64)
        baseline_model = baseline_model.to(device)
        
        # Train multiscale model
        print("\nTraining Multiscale Model...")
        train_losses = train_multiscale_model(
            multiscale_model,
            train_data,
            num_epochs=30,
            batch_size=32
        )
        
        # Train baseline model (simple training on 1H data)
        print("\nTraining Baseline Model...")
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
        baseline_criterion = nn.MSELoss()
        
        for epoch in range(30):
            batch_size = 32
            data_1h = train_data['1H']
            
            for i in range(0, len(data_1h) - batch_size - 1, batch_size):
                inputs = torch.FloatTensor(data_1h[i:i+batch_size]).to(device)
                targets = torch.FloatTensor(
                    data_1h[i+1:i+batch_size+1, 0]
                ).unsqueeze(-1).to(device)
                
                predictions = baseline_model(inputs)
                loss = baseline_criterion(predictions, targets)
                
                baseline_optimizer.zero_grad()
                loss.backward()
                baseline_optimizer.step()
        
        # Evaluate
        print("\nEvaluating models...")
        eval_results = evaluate_models(
            multiscale_model,
            baseline_model,
            test_data,
            device
        )
        
        if eval_results:
            results[asset] = eval_results
            
            print(f"\n{asset} Results:")
            print(f"  Multiscale MSE: {eval_results['multiscale_mse']:.4f}")
            print(f"  Baseline MSE:   {eval_results['baseline_mse']:.4f}")
            print(f"  Improvement:    {eval_results['improvement_pct']:.2f}%")
    
    # Save results
    output_dir = Path("outputs/multiscale_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp1_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Summary
    print("\nSummary:")
    for asset, result in results.items():
        print(f"  {asset}: {result['improvement_pct']:.2f}% improvement over baseline")
    
    avg_improvement = np.mean([r['improvement_pct'] for r in results.values()])
    print(f"\nAverage improvement: {avg_improvement:.2f}%")
    
    success = avg_improvement >= 10.0
    status = "✅ SUCCESS" if success else "⚠️  BELOW TARGET"
    print(f"\n{status} (Target: +10-15% improvement)")


if __name__ == "__main__":
    main()
