"""
SNN Interface Theory
SNNs as interface between trader perception and market reality
Test Hoffman's "fitness beats truth" in trading context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

from models.snn_trading_agent import SpikingTradingAgent


class AccurateMarketModel(nn.Module):
    """Truth-seeking model: tries to accurately predict market."""
    
    def __init__(self, input_dim=50, hidden_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def load_mock_market_data(num_samples=1000, input_dim=50):
    """Generate mock market data."""
    np.random.seed(42)
    
    # Generate features
    features = np.random.randn(num_samples, input_dim) * 10 + 100
    
    # Generate prices with some structure
    prices = 100 + np.cumsum(np.random.randn(num_samples) * 0.5)
    
    # Calculate returns
    returns = np.diff(prices)
    returns = np.concatenate([[0], returns])
    
    return features, prices, returns


def train_fitness_optimized_snn(
    snn_agent: SpikingTradingAgent,
    features: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    num_epochs: int = 20
):
    """Train SNN to maximize trading fitness (profit)."""
    print("\nTraining fitness-optimized SNN...")
    
    optimizer = torch.optim.Adam(snn_agent.parameters(), lr=0.001)
    
    batch_size = 32
    
    for epoch in range(num_epochs):
        epoch_fitness = 0
        num_batches = 0
        
        for i in range(0, len(features) - batch_size, batch_size):
            inputs = torch.FloatTensor(features[i:i+batch_size]).to(device)
            batch_returns = returns[i:i+batch_size]
            
            # SNN output: trading signal
            outputs, _ = snn_agent(inputs, num_steps=10)
            
            # Convert to trading decisions (simplified)
            decisions = torch.tanh(outputs.squeeze())  # [-1, 1]
            
            # Calculate fitness: profit from following signals
            batch_returns_tensor = torch.FloatTensor(batch_returns).to(device)
            fitness = (decisions * batch_returns_tensor).mean()  # Aligned = profit
            
            # Maximize fitness (minimize negative fitness)
            loss = -fitness
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_fitness += fitness.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_fitness = epoch_fitness / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Fitness: {avg_fitness:.4f}")


def train_truth_seeking_model(
    model: AccurateMarketModel,
    features: np.ndarray,
    prices: np.ndarray,
    device: torch.device,
    num_epochs: int = 20
):
    """Train model to accurately predict market (truth-seeking)."""
    print("\nTraining truth-seeking model...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    batch_size = 32
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(features) - batch_size - 1, batch_size):
            inputs = torch.FloatTensor(features[i:i+batch_size]).to(device)
            targets = torch.FloatTensor(prices[i+1:i+batch_size+1]).unsqueeze(-1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], MSE: {avg_loss:.4f}")


def evaluate_trading_performance(
    model: nn.Module,
    features: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    is_snn: bool = False
):
    """Evaluate actual trading performance."""
    model.eval()
    
    with torch.no_grad():
        inputs = torch.FloatTensor(features).to(device)
        
        if is_snn:
            outputs, _ = model(inputs, num_steps=10)
            decisions = torch.tanh(outputs.squeeze())
        else:
            predictions = model(inputs).squeeze()
            # Convert predictions to trading signals
            if len(predictions) > 1:
                price_changes = predictions[1:] - predictions[:-1]
                decisions = torch.tanh(price_changes)
                # Pad to match input length
                decisions = torch.cat([torch.zeros(1, device=device), decisions])
            else:
                decisions = torch.zeros_like(predictions)
        
        # Calculate cumulative returns
        returns_tensor = torch.FloatTensor(returns).to(device)
        strategy_returns = decisions * returns_tensor
        
        cumulative_return = strategy_returns.sum().item()
        sharpe_ratio = strategy_returns.mean().item() / (strategy_returns.std().item() + 1e-8)
        
    return {
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(features)
    }


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("SNN INTERFACE THEORY - Fitness vs Truth")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    num_samples = 1000
    
    # Load market data
    print("\nLoading market data...")
    features, prices, returns = load_mock_market_data(num_samples, input_dim)
    
    # Split train/test
    split_idx = int(0.8 * num_samples)
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    train_returns = returns[:split_idx]
    test_returns = returns[split_idx:]
    
    # Create models
    print("\nCreating models...")
    
    # Fitness-optimized SNN
    fitness_snn = SpikingTradingAgent(
        input_dim=input_dim,
        hidden_dim=100,
        output_dim=1,
        num_pathways=3
    ).to(device)
    
    # Truth-seeking model
    truth_model = AccurateMarketModel(
        input_dim=input_dim,
        hidden_dim=100
    ).to(device)
    
    # Train models
    train_fitness_optimized_snn(fitness_snn, train_features, train_returns, device)
    train_truth_seeking_model(truth_model, train_features, train_prices, device)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION - Trading Performance")
    print("=" * 80)
    
    fitness_results = evaluate_trading_performance(
        fitness_snn, test_features, test_returns, device, is_snn=True
    )
    
    truth_results = evaluate_trading_performance(
        truth_model, test_features, test_returns, device, is_snn=False
    )
    
    print("\nFitness-Optimized SNN:")
    print(f"  Cumulative Return: {fitness_results['cumulative_return']:.4f}")
    print(f"  Sharpe Ratio: {fitness_results['sharpe_ratio']:.4f}")
    
    print("\nTruth-Seeking Model:")
    print(f"  Cumulative Return: {truth_results['cumulative_return']:.4f}")
    print(f"  Sharpe Ratio: {truth_results['sharpe_ratio']:.4f}")
    
    # Determine winner
    fitness_wins = fitness_results['sharpe_ratio'] > truth_results['sharpe_ratio']
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    
    if fitness_wins:
        print("\n✅ FITNESS BEATS TRUTH")
        print("Fitness-optimized SNN outperforms truth-seeking model")
        print("Supports Hoffman's interface theory in trading context")
    else:
        print("\n⚠️  TRUTH BEATS FITNESS")
        print("Truth-seeking model outperforms fitness-optimized SNN")
        print("Challenges direct application of interface theory")
    
    # Save results
    results = {
        'fitness_optimized': {
            'cumulative_return': float(fitness_results['cumulative_return']),
            'sharpe_ratio': float(fitness_results['sharpe_ratio'])
        },
        'truth_seeking': {
            'cumulative_return': float(truth_results['cumulative_return']),
            'sharpe_ratio': float(truth_results['sharpe_ratio'])
        },
        'fitness_wins': bool(fitness_wins)
    }
    
    output_dir = Path("outputs/market_consciousness")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"snn_interface_theory_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
