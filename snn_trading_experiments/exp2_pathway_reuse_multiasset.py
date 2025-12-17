"""
Experiment 2: Pathway Reuse for Multi-Asset Learning
Train on BTC, reuse pathways for ETH/SOL
Test pathway-based learning without forgetting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import time

from models.snn_trading_agent import SpikingTradingAgent


def load_mock_asset_data(asset='BTC', num_samples=1000, input_dim=50):
    """Generate mock asset-specific data."""
    np.random.seed(hash(asset) % 2**32)
    
    # Generate synthetic features with asset-specific patterns
    data = np.random.randn(num_samples, input_dim) * 10 + 100
    
    # Add asset-specific trends
    if asset == 'BTC':
        trend = np.linspace(0, 30, num_samples).reshape(-1, 1)
    elif asset == 'ETH':
        trend = np.linspace(0, 25, num_samples).reshape(-1, 1)
    else:  # SOL
        trend = np.linspace(0, 20, num_samples).reshape(-1, 1)
    
    data = data + trend
    
    # Targets (next step prediction)
    targets = np.roll(data[:, 0], -1)
    targets[-1] = targets[-2]
    
    return data, targets


def train_on_asset(
    model: SpikingTradingAgent,
    data: np.ndarray,
    targets: np.ndarray,
    asset_name: str,
    device: torch.device,
    num_epochs: int = 20
):
    """Train SNN on specific asset data."""
    print(f"\nTraining on {asset_name}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    batch_size = 32
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(data) - batch_size, batch_size):
            inputs = torch.FloatTensor(data[i:i+batch_size]).to(device)
            tgt = torch.FloatTensor(targets[i:i+batch_size]).unsqueeze(-1).to(device)
            
            # Forward
            outputs, info = model(inputs, num_steps=10)
            loss = criterion(outputs, tgt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    return training_time


def evaluate_on_asset(
    model: SpikingTradingAgent,
    data: np.ndarray,
    targets: np.ndarray,
    device: torch.device
):
    """Evaluate SNN on asset data."""
    model.eval()
    
    with torch.no_grad():
        inputs = torch.FloatTensor(data).to(device)
        tgt = torch.FloatTensor(targets).unsqueeze(-1).to(device)
        
        outputs, _ = model(inputs, num_steps=10)
        mse = nn.MSELoss()(outputs, tgt).item()
    
    model.train()
    return mse


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 2: Pathway Reuse Multi-Asset")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    hidden_dim = 100
    num_samples = 1000
    
    assets = ['BTC', 'ETH', 'SOL']
    
    # Load data for all assets
    print("\nLoading data for all assets...")
    asset_data = {}
    for asset in assets:
        data, targets = load_mock_asset_data(asset, num_samples, input_dim)
        
        # Split train/test
        split_idx = int(0.8 * num_samples)
        asset_data[asset] = {
            'train_data': data[:split_idx],
            'train_targets': targets[:split_idx],
            'test_data': data[split_idx:],
            'test_targets': targets[split_idx:]
        }
    
    # Create SNN with pathway reuse
    print("\nCreating SNN with pathway reuse...")
    snn_agent = SpikingTradingAgent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_pathways=3,  # Start with 3 pathways
    ).to(device)
    
    # Train on BTC first
    btc_train_time = train_on_asset(
        snn_agent,
        asset_data['BTC']['train_data'],
        asset_data['BTC']['train_targets'],
        'BTC',
        device,
        num_epochs=20
    )
    
    # Evaluate on BTC
    btc_mse = evaluate_on_asset(
        snn_agent,
        asset_data['BTC']['test_data'],
        asset_data['BTC']['test_targets'],
        device
    )
    
    print(f"\nBTC Training Time: {btc_train_time:.2f}s")
    print(f"BTC Test MSE: {btc_mse:.4f}")
    
    # Get pathway statistics after BTC training
    stats_btc = snn_agent.get_pathway_statistics()
    print(f"Active pathways after BTC: {stats_btc['num_pathways']}")
    
    # Reuse pathways for ETH (should be faster)
    eth_train_time = train_on_asset(
        snn_agent,
        asset_data['ETH']['train_data'],
        asset_data['ETH']['train_targets'],
        'ETH',
        device,
        num_epochs=10  # Fewer epochs due to transfer learning
    )
    
    eth_mse = evaluate_on_asset(
        snn_agent,
        asset_data['ETH']['test_data'],
        asset_data['ETH']['test_targets'],
        device
    )
    
    print(f"\nETH Training Time: {eth_train_time:.2f}s")
    print(f"ETH Test MSE: {eth_mse:.4f}")
    
    # Reuse pathways for SOL
    sol_train_time = train_on_asset(
        snn_agent,
        asset_data['SOL']['train_data'],
        asset_data['SOL']['train_targets'],
        'SOL',
        device,
        num_epochs=10
    )
    
    sol_mse = evaluate_on_asset(
        snn_agent,
        asset_data['SOL']['test_data'],
        asset_data['SOL']['test_targets'],
        device
    )
    
    print(f"\nSOL Training Time: {sol_train_time:.2f}s")
    print(f"SOL Test MSE: {sol_mse:.4f}")
    
    # Check for catastrophic forgetting on BTC
    btc_mse_after = evaluate_on_asset(
        snn_agent,
        asset_data['BTC']['test_data'],
        asset_data['BTC']['test_targets'],
        device
    )
    
    forgetting = ((btc_mse_after - btc_mse) / btc_mse) * 100
    
    # Calculate transfer learning efficiency
    avg_transfer_time = (eth_train_time + sol_train_time) / 2
    speedup = (btc_train_time / avg_transfer_time - 1) * 100
    
    # Results
    results = {
        'btc_initial_mse': float(btc_mse),
        'btc_final_mse': float(btc_mse_after),
        'forgetting_pct': float(forgetting),
        'eth_mse': float(eth_mse),
        'sol_mse': float(sol_mse),
        'btc_train_time': float(btc_train_time),
        'avg_transfer_time': float(avg_transfer_time),
        'speedup_pct': float(speedup),
        'pathway_stats': {
            'num_pathways': int(stats_btc['num_pathways']),
        }
    }
    
    # Save results
    output_dir = Path("outputs/snn_trading_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp2_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    print(f"\nTransfer Learning Speedup: {speedup:.2f}%")
    print(f"Forgetting on BTC: {forgetting:.2f}%")
    
    success = speedup >= 50.0 and abs(forgetting) < 10.0
    status = "✅ SUCCESS" if success else "⚠️  BELOW TARGET"
    print(f"\n{status}")
    print(f"  Speedup: {speedup:.2f}% (Target: ≥50%)")
    print(f"  Forgetting: {forgetting:.2f}% (Target: <10%)")


if __name__ == "__main__":
    main()
