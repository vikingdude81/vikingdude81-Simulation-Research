"""
Experiment 3: Dynamic Growth Adaptation
Start with small SNN (100 neurons), grow based on market complexity
Track performance during growth
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


def load_mock_data_with_complexity(num_samples=2000, input_dim=50):
    """Generate data with varying complexity over time."""
    np.random.seed(42)
    
    # First half: simple linear trend
    simple_data = np.random.randn(num_samples // 2, input_dim) * 5 + 100
    simple_trend = np.linspace(0, 10, num_samples // 2).reshape(-1, 1)
    simple_data = simple_data + simple_trend
    
    # Second half: complex non-linear pattern
    complex_data = np.random.randn(num_samples // 2, input_dim) * 15 + 120
    t = np.linspace(0, 4*np.pi, num_samples // 2)
    complex_trend = (10 * np.sin(t) + 0.5 * t).reshape(-1, 1)
    complex_data = complex_data + complex_trend
    
    # Combine
    data = np.vstack([simple_data, complex_data])
    
    # Targets
    targets = np.roll(data[:, 0], -1)
    targets[-1] = targets[-2]
    
    return data, targets


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("EXPERIMENT 3: Dynamic Growth Adaptation")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    initial_hidden_dim = 100  # Start small
    num_samples = 2000
    
    # Load data with varying complexity
    print("\nLoading data with varying complexity...")
    data, targets = load_mock_data_with_complexity(num_samples, input_dim)
    
    # Create SNN with small initial size
    print(f"\nCreating SNN with initial hidden_dim={initial_hidden_dim}...")
    snn_agent = SpikingTradingAgent(
        input_dim=input_dim,
        hidden_dim=initial_hidden_dim,
        output_dim=1,
        num_pathways=2,  # Start with just 2 pathways
        growth_threshold=0.85  # Grow if performance below 85%
    ).to(device)
    
    optimizer = torch.optim.Adam(snn_agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training with dynamic growth
    batch_size = 32
    num_epochs = 50
    window_size = 500  # Evaluate every window_size samples
    
    performance_history = []
    growth_events = []
    
    print("\nTraining with dynamic growth monitoring...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(data) - batch_size, batch_size):
            inputs = torch.FloatTensor(data[i:i+batch_size]).to(device)
            tgt = torch.FloatTensor(targets[i:i+batch_size]).unsqueeze(-1).to(device)
            
            # Forward
            outputs, info = snn_agent(inputs, num_steps=10)
            loss = criterion(outputs, tgt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Calculate performance metric (inverse of loss, normalized)
        performance = 1.0 / (1.0 + avg_loss)
        performance_history.append(performance)
        
        # Check if should grow
        if snn_agent.should_grow(performance):
            print(f"\n  🌱 Growing network at epoch {epoch+1}")
            new_pathway_id = snn_agent.add_pathway()
            growth_events.append({
                'epoch': epoch + 1,
                'pathway_id': new_pathway_id,
                'performance': performance,
                'num_pathways': len(snn_agent.pathways)
            })
            
            # Re-create optimizer to include new parameters
            optimizer = torch.optim.Adam(snn_agent.parameters(), lr=0.001)
        
        if (epoch + 1) % 10 == 0:
            stats = snn_agent.get_pathway_statistics()
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                  f"Pathways: {stats['num_pathways']}, Performance: {performance:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    snn_agent.eval()
    
    with torch.no_grad():
        # Evaluate on full dataset
        test_inputs = torch.FloatTensor(data).to(device)
        test_targets = torch.FloatTensor(targets).unsqueeze(-1).to(device)
        
        test_outputs, _ = snn_agent(test_inputs, num_steps=10)
        final_mse = nn.MSELoss()(test_outputs, test_targets).item()
    
    # Calculate improvement from growth
    if len(performance_history) > 10:
        initial_avg_perf = np.mean(performance_history[:10])
        final_avg_perf = np.mean(performance_history[-10:])
        improvement = ((final_avg_perf - initial_avg_perf) / initial_avg_perf) * 100
    else:
        improvement = 0.0
    
    # Results
    results = {
        'initial_hidden_dim': initial_hidden_dim,
        'final_num_pathways': len(snn_agent.pathways),
        'num_growth_events': len(growth_events),
        'growth_events': growth_events,
        'final_mse': float(final_mse),
        'performance_improvement_pct': float(improvement),
        'performance_history': [float(p) for p in performance_history]
    }
    
    # Save results
    output_dir = Path("outputs/snn_trading_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exp3_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    print(f"\nGrowth Summary:")
    print(f"  Initial pathways: 2")
    print(f"  Final pathways: {results['final_num_pathways']}")
    print(f"  Growth events: {results['num_growth_events']}")
    print(f"  Performance improvement: {improvement:.2f}%")
    
    success = improvement >= 20.0
    status = "✅ SUCCESS" if success else "⚠️  BELOW TARGET"
    print(f"\n{status} - Performance improvement: {improvement:.2f}% (Target: ≥20%)")


if __name__ == "__main__":
    main()
