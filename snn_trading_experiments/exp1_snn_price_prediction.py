"""
Experiment 1: SNN Price Prediction
Replace LSTM with SNN for price forecasting
Encode price as spike rates and compare accuracy
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

from models.snn_trading_agent import SpikingTradingAgent
from utils.snn_utils import encode_price_movements, decode_from_spikes


class SimpleLSTM(nn.Module):
    """Baseline LSTM model for comparison."""
    
    def __init__(self, input_dim=50, hidden_dim=100, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred


def load_mock_price_data(num_samples=1000, num_features=50):
    """Generate mock price data for testing."""
    np.random.seed(42)
    
    # Generate synthetic price series
    prices = 100 + np.cumsum(np.random.randn(num_samples) * 0.5)
    
    # Generate features (technical indicators, etc.)
    features = np.random.randn(num_samples, num_features) * 10
    
    # Add price-based features
    features[:, 0] = prices
    features[:, 1] = np.gradient(prices)  # Price change
    
    return features, prices


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 1: SNN Price Prediction")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    hidden_dim = 100
    num_samples = 1000
    
    # Load data
    print("\nLoading price data...")
    features, prices = load_mock_price_data(num_samples, input_dim)
    
    # Split train/test
    split_idx = int(0.8 * num_samples)
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    
    # Create models
    print("\nCreating models...")
    
    snn_agent = SpikingTradingAgent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,  # Single price prediction
        num_pathways=3
    ).to(device)
    
    lstm_model = SimpleLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1
    ).to(device)
    
    # Train SNN
    print("\nTraining SNN...")
    snn_optimizer = torch.optim.Adam(snn_agent.parameters(), lr=0.001)
    snn_criterion = nn.MSELoss()
    
    batch_size = 32
    num_epochs = 20
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_features) - batch_size, batch_size):
            inputs = torch.FloatTensor(
                train_features[i:i+batch_size]
            ).to(device)
            
            targets = torch.FloatTensor(
                train_prices[i+1:i+batch_size+1]
            ).unsqueeze(-1).to(device)
            
            # Forward pass
            outputs, info = snn_agent(inputs, num_steps=10)
            
            # Loss
            loss = snn_criterion(outputs, targets)
            
            # Backward
            snn_optimizer.zero_grad()
            loss.backward()
            snn_optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Train LSTM
    print("\nTraining LSTM...")
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for i in range(0, len(train_features) - batch_size, batch_size):
            inputs = torch.FloatTensor(
                train_features[i:i+batch_size]
            ).unsqueeze(1).to(device)  # Add sequence dimension
            
            targets = torch.FloatTensor(
                train_prices[i+1:i+batch_size+1]
            ).unsqueeze(-1).to(device)
            
            outputs = lstm_model(inputs)
            loss = lstm_criterion(outputs, targets)
            
            lstm_optimizer.zero_grad()
            loss.backward()
            lstm_optimizer.step()
    
    # Evaluate
    print("\nEvaluating models...")
    
    snn_agent.eval()
    lstm_model.eval()
    
    with torch.no_grad():
        # Test on test set
        test_inputs = torch.FloatTensor(test_features).to(device)
        test_targets = torch.FloatTensor(test_prices[1:]).unsqueeze(-1).to(device)
        
        # SNN predictions
        snn_outputs, _ = snn_agent(test_inputs[:-1], num_steps=10)
        snn_mse = nn.MSELoss()(snn_outputs, test_targets).item()
        
        # LSTM predictions
        lstm_inputs = test_inputs[:-1].unsqueeze(1)
        lstm_outputs = lstm_model(lstm_inputs)
        lstm_mse = nn.MSELoss()(lstm_outputs, test_targets).item()
    
    # Results
    results = {
        'snn_mse': float(snn_mse),
        'lstm_mse': float(lstm_mse),
        'snn_better': snn_mse <= lstm_mse,
        'relative_performance': float((lstm_mse - snn_mse) / lstm_mse * 100)
    }
    
    # Save results
    output_dir = Path("outputs/snn_trading_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp1_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    print(f"\nSNN MSE:  {snn_mse:.4f}")
    print(f"LSTM MSE: {lstm_mse:.4f}")
    
    if results['snn_better']:
        print(f"\n✅ SUCCESS - SNN outperforms LSTM by {results['relative_performance']:.2f}%")
    else:
        print(f"\n⚠️  SNN underperforms LSTM by {-results['relative_performance']:.2f}%")


if __name__ == "__main__":
    main()
